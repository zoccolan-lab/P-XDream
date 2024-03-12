from copy import deepcopy

import pandas as pd
from script.MaximizeActivity.plots import plot_optimizing_units, plot_scores, plot_scores_by_cat
from zdream.experiment import Experiment, MultiExperiment
from zdream.generator import Generator, InverseAlexGenerator
from zdream.logger import Logger, LoguruLogger
from zdream.optimizer import GeneticOptimizer, Optimizer
from zdream.probe import RecordingProbe
from zdream.scores import MaxActivityScorer, Scorer
from zdream.subject import InSilicoSubject, NetworkSubject
from zdream.utils.dataset import MiniImageNet
from zdream.utils.io_ import to_gif
from zdream.utils.misc import concatenate_images, device
from zdream.utils.model import Codes, DisplayScreen, MaskGenerator, Message, Stimuli, StimuliScore, SubjectState, aggregating_functions, mask_generator_from_template
from zdream.utils.parsing import parse_boolean_string, parse_layer_target_units, parse_scoring_units

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import os
from os import path
from typing import Any, Dict, List, Tuple, Type, cast


class MaximizeActivityExperiment(Experiment):

    EXPERIMENT_TITLE = "MaximizeActivity"

    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'

    @property
    def scorer(self) -> MaxActivityScorer: return cast(MaxActivityScorer, self._scorer) 

    @classmethod
    def _from_config(cls, conf : Dict[str, Any]) -> 'MaximizeActivityExperiment':
        '''
        Static constructor for a _MaximizeActivityExperiment class from configuration file.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: _MaximizeActivityExperiment instance with hyperparameters set from configuration.
        :rtype: _MaximizeActivity
        '''

        # Extract specific configurations
        gen_conf = conf['generator']
        msk_conf = conf['mask_generator']
        sbj_conf = conf['subject']
        scr_conf = conf['scorer']
        opt_conf = conf['optimizer']
        log_conf = conf['logger']

        # --- MASK GENERATOR ---

        template = parse_boolean_string(boolean_str=msk_conf['template'])
        mask_generator = mask_generator_from_template(template=template, shuffle=msk_conf['shuffle'])
        
        # --- GENERATOR ---

        # Dataloader
        use_nat = template.count(False) > 0
        if use_nat:
            dataset    = MiniImageNet(root=gen_conf['mini_inet'])
            dataloader = DataLoader(dataset, batch_size=gen_conf['batch_size'], shuffle=True)

        # Instance
        generator = InverseAlexGenerator(
            root           = gen_conf['weights'],
            variant        = gen_conf['variant'],
            nat_img_loader = dataloader if use_nat else None
        ).to(device)


        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = NetworkSubject(network_name=sbj_conf['net_name']).layer_info

        # Probe
        record_target = parse_layer_target_units(input_str=sbj_conf['rec_layers'], net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore

        # Subject with attached recording probe
        sbj_net = NetworkSubject(
            record_probe=probe,
            network_name=sbj_conf['net_name']
        )
        
        sbj_net._network.eval() # TODO cannot access private attribute, make public method to call the eval

        # --- SCORER ---

        # Target neurons
        #TODO: use rec_only dictionary to index exp.subject.states_history
        score_dict, rec_only  = parse_scoring_units(
            input_str=scr_conf['scr_layers'], 
            net_info=layer_info,
            rec_neurons=record_target
        )

        scorer = MaxActivityScorer(
            trg_neurons=score_dict,
            aggregate=aggregating_functions[scr_conf['aggregation']]
        )

        # --- OPTIMIZER ---

        optim = GeneticOptimizer(
            states_shape   = generator.input_dim,
            random_seed    =     conf['random_seed'],
            random_distr   = opt_conf['random_distr'],
            mutation_rate  = opt_conf['mutation_rate'],
            mutation_size  = opt_conf['mutation_size'],
            population_size= opt_conf['pop_size'],
            temperature    = opt_conf['temperature'],
            num_parents    = opt_conf['num_parents']
        )

        #  --- LOGGER --- 

        log_conf['title'] = MaximizeActivityExperiment.EXPERIMENT_TITLE
        logger = LoguruLogger(conf=log_conf)
        
        # In the case render option is enabled we add display screens
        if conf['render']:

            # In the case of multi-experiment run, the shared screens
            # are set in `display_screens` entry
            if 'display_screens' in conf:
                for screen in conf['display_screens']:
                    logger.add_screen(screen=screen)

            # If the key is not set it is the case of a single experiment
            # and we create screens instance
            else:

                # Add screen for synthetic images
                logger.add_screen(
                    screen=DisplayScreen(title=cls.GEN_IMG_SCREEN, display_size=(400, 400))
                )

                # Add screen fro natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=(400, 400))
                    )

        # --- DATA ---
        data = {
            "dataset": dataset if use_nat else None,
            'display_plots': conf['display_plots'],
            'render': conf['render'],
            'close_screen': conf.get('close_screen', False),
            'rec_only': rec_only
        }

        # Experiment configuration
        experiment = cls(
            generator=generator,
            scorer=scorer,
            optimizer=optim,
            subject=sbj_net,
            logger=logger,
            iteration=conf['iter'],
            mask_generator=mask_generator,
            data=data, name=log_conf['name']
        )

        return experiment

    def __init__(
        self,    
        generator:      Generator,
        subject:        InSilicoSubject,
        scorer:         Scorer,
        optimizer:      Optimizer,
        iteration:      int,
        logger:         Logger,
        mask_generator: MaskGenerator | None = None,
        data:           Dict[str, Any] = dict(),
        name:           str = 'experiment'
    ) -> None:

        super().__init__(
            generator      = generator,
            subject        = subject,
            scorer         = scorer,
            optimizer      = optimizer,
            iteration      = iteration,
            logger         = logger,
            mask_generator = mask_generator,
            data           = data,
            name           = name,
        )

        # Create a mock mask for one synthetic image 
        # to see if natural images are involved in the experiment
        mock_mask = self._mask_generator(1)
    
        self._use_natural = mock_mask is not None and mock_mask.count(False) > 0

        # Extract from Data

        # Visualization flags
        self._display_plots = cast(bool, data['display_plots'])
        self._render        = cast(bool, data['render'])
        self._close_screen  = cast(bool, data['close_screen'])
        
        self._rec_only      = cast(Dict[str, List[int]| None] , data['rec_only'])

        if self._use_natural:
            self._dataset   = cast(MiniImageNet, data['dataset'])

    def _progress_info(self, i: int) -> str:

        # We add the progress information about the best
        # and the average score per each iteration
        stat_gen = self.optimizer.stats

        if self._use_natural:
            stat_nat = self.optimizer.stats_nat

        best_gen = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen = cast(NDArray, stat_gen['curr_score']).mean()
        if self._use_natural:
            best_nat = cast(NDArray, stat_nat['best_score']).mean()

        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        if self._use_natural:
            best_nat_str = f'{best_nat:.1f}'

        if self._use_natural:
            desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str} | best nat: {best_nat_str}'
        else:
            desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'

        progress_super = super()._progress_info(i=i)

        return f'{progress_super}{desc}'

    def _init(self):

        super()._init()

        # Data structure to save best score and best image
        if self._use_natural:
            self._best_nat_scr = 0
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
        
        # Set gif
        self._gif: List[Image.Image] = []

        # Last seen labels
        if self._use_natural:
            self._labels: List[int] = []


    def _progress(self, i: int):

        super()._progress(i)

        # Get best stimuli
        best_code = self.optimizer.solution
        best_synthetic, _ = self.generator(codes=best_code, pipeline=False)
        best_synthetic_img = to_pil_image(best_synthetic[0])

        if self._use_natural:
            best_natural = self._best_nat_img
            
        if not self._gif or self._gif[-1] != best_synthetic_img:
            self._gif.append(
                best_synthetic_img
            )
            
        if self._render:

            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )

            if self._use_natural:
                self._logger.update_screen(
                    screen_name=self.NAT_IMG_SCREEN,
                    image=to_pil_image(best_natural)
                )

        

    def _finish(self):

        super()._finish()

        # Close screens
        if self._close_screen:
            self._logger.close_all_screens()

        # 1. Save visual stimuli (synthetic and natural)

        img_dir = path.join(self.target_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        self._logger.info(mess=f"Saving images to {img_dir}")

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen, _ = self.generator(codes=self.optimizer.solution, pipeline=False)
        best_gen = best_gen[0] # remove 1 batch size

        # We retrieve the stored best natural image
        if self._use_natural:
            best_nat = self._best_nat_img

        # Saving images
        for img, label in [
            (to_pil_image(best_gen), 'best synthetic'),
            (to_pil_image(best_nat), 'best natural'),
            (concatenate_images(img_list=[best_gen, best_nat]), 'best stimuli'),
        ] if self._use_natural else [
            (to_pil_image(best_gen), 'best synthetic')
        ]:
            out_fp = path.join(img_dir, f'{label.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {label} image to {out_fp}')
            img.save(out_fp)
        
        out_fp = path.join(img_dir, 'evolving_best.gif')
        self._logger.info(f'> Saving evolving best stimuli across generations to {out_fp}')
        to_gif(image_list=self._gif, out_fp=out_fp)

        self._logger.info(mess='')
        
        # 2. Save plots

        plots_dir = path.join(self.target_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        self._logger.info(mess=f"Saving plots to {plots_dir}")
        
        self._logger.prefix='> '
        plot_scores(
            scores=(
                self.optimizer.scores_history,
                self.optimizer.scores_nat_history if self._use_natural else np.array([])
            ),
            stats=(
                self.optimizer.stats,
                self.optimizer.stats_nat if self._use_natural else dict(),
            ),
            out_dir=plots_dir,
            display_plots=self._display_plots,
            logger=self._logger
        )

        if self._use_natural:
            plot_scores_by_cat(
                scores=(
                    self.optimizer.scores_history,
                    self.optimizer.scores_nat_history
                ),
                lbls    = self._labels,
                out_dir = plots_dir, 
                dataset = self._dataset,
                display_plots=self._display_plots,
                logger=self._logger
            )
        self._logger.prefix=''
        
        self._logger.info(mess='')
        

    def _stimuli_to_sbj_state(self, data: Tuple[Stimuli, Message]) -> Tuple[SubjectState, Message]:

        # We save the last set of stimuli
        self._stimuli, msg = data
        if self._use_natural:
            self._labels.extend(msg.label)

        return super()._stimuli_to_sbj_state(data)

    def _stm_score_to_codes(self, data: Tuple[StimuliScore, Message]) -> Codes:

        sub_score, msg = data

        # We inspect if the new set of stimuli (both synthetic and natural)
        # achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_natural:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._stm_score_to_codes((sub_score, msg))
    
    
class NeuronScoreMultiExperiment(MultiExperiment):

    def __init__(
            self, 
            experiment:      Type['Experiment'], 
            experiment_conf: Dict[str, List[Any]], 
            default_conf:    Dict[str, Any]
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)

        # Add the close screen flag to the last configuration
        self._search_config[-1]['close_screen'] = True
    
    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [
            DisplayScreen(title=MaximizeActivityExperiment.GEN_IMG_SCREEN, display_size=(400, 400))
        ]

        # Add screen for natural images if at least one will use it
        if any(parse_boolean_string(conf['mask_generator']['template']).count(False) > 0 for conf in self._search_config):
            screens.append(
                DisplayScreen(title=MaximizeActivityExperiment.NAT_IMG_SCREEN, display_size=(400, 400))
            )

        return screens
        
    def _init(self):
        super()._init()
        
        self._data['desc']      = 'Scores at varying number of scoring neurons'
        self._data['score']         = list()
        self._data['neurons']       = list()
        self._data['layer']         = list()
        self._data['deltaA_rec']    = list()

        self._data['Copt_rec']      = list()
        ''' 
        Correlation between the optimized activation as a 
        one-dimensional array with the length the number of iterations.
        '''

        self._data['Copt_rec_var']  = list()
        self._data['Crec_rec']      = list()
        self._data['Crec_rec_var']  = list()
        self._data['num_gens']      = list()
        self._data['desc']      = 'Scores at varying number of scoring neurons'
        self._data['score']         = list()
        self._data['neurons']       = list()
        self._data['layer']         = list()
        self._data['deltaA_rec']    = list()

        self._data['Copt_rec']      = list()
        ''' 
        Correlation between the optimized activation as a 
        one-dimensional array with the length the number of iterations.
        '''

        self._data['Copt_rec_var']  = list()
        self._data['Crec_rec']      = list()
        self._data['Crec_rec_var']  = list()
        self._data['num_gens']      = list()

    @property
    def _logger_type(self) -> Type[Logger]:
        return LoguruLogger

    def _progress(self, exp: MaximizeActivityExperiment, i: int):
        super()._progress(exp, i)

        self._data['score']  .append(exp.optimizer.stats['best_score'])
        
        deltaA_rec={}; Copt_rec={}; Copt_rec_var={}
        Crec_rec={}; Crec_rec_var={}
        #iterate over recorded layers to get statistics about non optimized sites
        for k,v in exp.subject.states_history.items():
            if exp._rec_only[k]: #if there are recording only neurons
                #get the average difference in activation between the last and the first optim iteration
                deltaA_rec[k] = np.mean(v[-1,:, exp._rec_only[k]]) - np.mean(v[0,:, exp._rec_only[k]])
                #compute the correlation between each avg recorded unit and the mean of the optimized
                # sites. Store both mean corr and its variance
                avg_rec = np.mean(v[:,:, exp._rec_only[k]],axis=1).T
                corr_vec = np.corrcoef(exp.optimizer.stats['mean_shist'], avg_rec)[0, 1:]
                Copt_rec[k] = np.mean(corr_vec)
                Copt_rec_var[k] = np.std(corr_vec)
                #compute the correlation between avg recorded units. Store both mean corr and its variance
                Crec_rec_mat = np.corrcoef(avg_rec)
                up_tria_idxs = np.triu_indices(Crec_rec_mat.shape[0], k=1)
                Crec_rec[k]  = np.mean(Crec_rec_mat[up_tria_idxs])
                Crec_rec_var[k] = np.std(Crec_rec_mat[up_tria_idxs])

        self._data['deltaA_rec']  .append(deltaA_rec)
        self._data['Copt_rec']  .append(Copt_rec)
        self._data['Copt_rec_var']  .append(Copt_rec_var)
        self._data['Crec_rec']  .append(Crec_rec)
        self._data['Crec_rec_var']  .append(Crec_rec_var)
        
        self._data['Copt_rec']  .append(Copt_rec)
        self._data['Copt_rec_var']  .append(Copt_rec_var)
        self._data['Crec_rec']  .append(Copt_rec)
        self._data['Crec_rec_var']  .append(Copt_rec_var)
        
        self._data['neurons'].append(exp.scorer.optimizing_units)
        self._data['layer']  .append(list(exp.scorer._trg_neurons.keys()))
        self._data['num_gens'].append(exp._iteration) # TODO make public property
    
        
    def _get_multiexp_df(self):
        """get a dataframe with the multiexperiment results

        :param multiexp_data: data attribute of NeuronScoreMultiExperiment
        :type multiexp_data: dict[str,Any]
        """
        
        multiexp_data = self._data
        def extract_layer(el):
            ''' Auxiliar function to extract layer names from a list element'''
            if len(el) > 1:
                raise ValueError(f'Expected to record from a single layer, multiple where found: {el}')
            return el[0]
        
        #extract data from multiexp_data as numpy arrays
        data = {
            'layers'  : np.stack([extract_layer(el) for el in multiexp_data['layer']]),
            'neurons' : np.stack(multiexp_data['neurons'], dtype=np.int32),
            'num_gens': np.stack(multiexp_data['num_gens'], dtype=np.int32),
            'scores'  : np.concatenate(multiexp_data['score'])    
        }


        
        #extract recording-related data 
        unique_keys = list(set().union(*[d.keys() for d in multiexp_data['deltaA_rec']]))
        for K in multiexp_data.keys():
            if '_rec' in K:
                prefix = 'Δrec_' if K=='deltaA_rec' else K
                rec_delta_dict = {prefix+key: np.full(len(multiexp_data[K]), 
                                np.nan, dtype=np.float32) for key in unique_keys}

                for i, d in enumerate(multiexp_data[K]):
                    for uk in unique_keys:
                        if uk in d.keys():
                            rec_delta_dict[prefix+uk][i] = d[uk]
                #unify all data into a single dictionary
                data.update(rec_delta_dict)
        #unique_keys are the unique sessions present in the dicts of multiexp_data rec-related data
        unique_keys = list(set().union(*[d.keys() for d in multiexp_data['deltaA_rec']]))
        for K in multiexp_data.keys():
            if '_rec' in K: #only for recording-related keys...
                prefix = 'Δrec_' if K=='deltaA_rec' else K
                #initialize the dict that will contain recording-related data 
                #as a np array full of nans
                rec_dict = {prefix+key: np.full(len(multiexp_data[K]), 
                                np.nan, dtype=np.float32) for key in unique_keys}
                #for each experiment, only if a recording of the layer of interest 
                # is present add it to the rec_dict
                for i, d in enumerate(multiexp_data[K]):
                    for uk in unique_keys:
                        if uk in d.keys():
                            rec_dict[prefix+uk][i] = d[uk]
                            
                #unify all data into a single dictionary
                data.update(rec_dict)
        
        self.out_df = pd.DataFrame(data)
        self.out_df.to_excel(path.join(self._logger.target_dir,'mrun_df.xlsx'), index=False)
        

    def _finish(self):
        self._get_multiexp_df()
        
        multiexp_lineplot(self.out_df, gr_vars= ['layers', 'neurons'], out_dir=self._logger.target_dir,
                      y_var = 'scores', metrics = ['mean', 'sem'],logger=self._logger, display_plots = False)
        
        for c in self.out_df.columns:
            if 'rec' in c:    
                multiexp_lineplot(self.out_df, gr_vars= ['layers', 'neurons'], out_dir=self._logger.target_dir,
                            y_var = c, metrics = ['mean', 'sem'],logger=self._logger, display_plots = False)
        
        super()._finish()




