from __future__ import annotations

import os
from os import path
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from typing import Any, Callable, Dict, List, Literal, Tuple
from analysis.utils.settings import CLU_ORDER, COLORS, LAYER_SETTINGS, WORDNET_DIR
from analysis.utils.wordnet import ImageNetWords, WordNet
from zdream.clustering.cluster import Clusters
from zdream.clustering.ds import DSClusters
from zdream.utils.io_ import load_pickle, read_json
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger
from zdream.utils.misc import default

# --- LOGGING ---

def start(logger: Logger, name: str):
    
    logger.info(mess=name)
    logger.formatting = lambda x: f'> {x}'

def end(logger: Logger):
    
    logger.info(mess='Done')
    logger.reset_formatting()
    logger.info(mess='')
    
# --- LOADING ---

SuperStimuliInfoName = Literal['code', 'fitness']
SuperStimuliInfo = Dict[SuperStimuliInfoName, Any]

class AlexNetLayerLoader:

    def __init__(self, alexnet_dir: str, layer: str, logger: Logger = SilentLogger()):

        self.alexnet_dir = alexnet_dir
        self.layer = layer
        self._logger = logger

    def __str__(self):
        return f"AlexNetInfoLoader(alexnet_dir={self.alexnet_dir}, layer={self.layer})"
    
    def __repr__(self):
        return str(self)
    
    @property
    def layer_dir(self):
        return os.path.join(self.alexnet_dir, LAYER_SETTINGS[self.layer]['directory'])
    
    @property
    def cluster_dir(self):
        return os.path.join(self.layer_dir, 'clusters')
    
    @property
    def superstimuli_dir(self):
        return os.path.join(self.layer_dir, 'superstimuli')
    
    @property
    def recordings_dir(self):
        return os.path.join(self.layer_dir, 'recordings')
    
    @property
    def fm_clu_segmentation_dir(self):
        return os.path.join(self.layer_dir, 'fm_clu_segmentation')
    
    def load_clusters(self, exclude: List[str] = []) -> Dict[str, Clusters]:
    
        clusters = {}
        
        self._logger.info(mess=f'Loading clusters from {dir}')
        
        for root, _, files in os.walk(self.cluster_dir):
            
            for file in files:
                
                if file.endswith(".json") and file not in exclude:
                    
                    file_path = os.path.join(root, file)
                    self._logger.info(mess=f' > Loading {file}')
                    clu_algo = file.replace(".json", "")

                    C = DSClusters if clu_algo == 'DominantSet'else Clusters
                    clusters[clu_algo] = C.from_file(fp=file_path)
                    
                else:
                    self._logger.info(mess=f' > Skipping {file}')
                    
        clusters = dict(sorted(clusters.items(), key=lambda x: CLU_ORDER[x[0]]))

        return clusters
    
    def load_recordings(self, inet=False) -> NDArray:
        
        f_name = 'recordings_inet' if inet else 'recordings'

        fp = os.path.join(self.recordings_dir, f'{f_name}.npy')
        self._logger.info(mess=f'Loading recordings from {fp}')
        return np.load(fp)
    
    def load_superstimuli(self) -> Tuple[Dict[str, Dict[int, SuperStimuliInfo]], Dict[int, SuperStimuliInfo]]:

        clu_fp = path.join(self.superstimuli_dir, 'cluster_superstimuli.pkl')
        units_fp = path.join(self.superstimuli_dir, 'units_superstimuli.pkl')

        self._logger.info(mess=f'Loading cluster superstimuli from {clu_fp}')
        clu_supertimuli = load_pickle(clu_fp)

        self._logger.info(mess=f'Loading units superstimuli from {units_fp}')
        units_supertimuli = load_pickle(units_fp)

        return clu_supertimuli, units_supertimuli

    def load_norm_fun(self, gen_variant: str) -> CurveFitter:

        fp = path.join(self.alexnet_dir, 'neuron_scaling', 'neuron_scaling_functions.pkl')

        self._logger.info(mess=f'Loading neuron scaling functions from {fp}')
        fun = load_pickle(fp)

        return fun[gen_variant][self.layer]
    
    def load_segmentation_info(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:

        fm_file  = path.join(self.fm_clu_segmentation_dir, 'fm_segmentation.json')
        clu_file = path.join(self.fm_clu_segmentation_dir, 'clu_segmentation.json')

        self._logger.info(mess=f'Loading feature map segmentation from {fm_file}')
        fm_segmentation = read_json(fm_file)

        self._logger.info(mess=f'Loading cluster segmentation from {clu_file}')
        clu_segmentation = read_json(clu_file)

        return fm_segmentation, clu_segmentation
    
    def load_segmentation_superstimuli(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:

        fm_file  = path.join(self.fm_clu_segmentation_dir, 'fm_segmentation_superstimuli.pkl')
        clu_file = path.join(self.fm_clu_segmentation_dir, 'clu_segmentation_superstimuli.pkl')

        self._logger.info(mess=f'Loading feature map segmentation from {fm_file}')
        fm_segmentation = load_pickle(fm_file)

        self._logger.info(mess=f'Loading cluster segmentation from {clu_file}')
        clu_segmentation = load_pickle(clu_file)

        return fm_segmentation, clu_segmentation

def load_wordnet(logger: Logger = SilentLogger()) -> WordNet:
    
    words_fp             = os.path.join(WORDNET_DIR, 'words.txt')
    hierarchy_fp         = os.path.join(WORDNET_DIR, 'wordnet.is_a.txt')
    words_precomputed_fp = os.path.join(WORDNET_DIR, 'words.pkl')
    
    # Load WordNet with precomputed words if available
    if os.path.exists(words_precomputed_fp):
        
        logger.info(mess='Loading precomputed WordNet')
        
        wordnet = WordNet.from_precomputed(
            wordnet_fp=words_fp, 
            hierarchy_fp=hierarchy_fp, 
            words_precomputed=words_precomputed_fp,
            logger=logger
        )
    
    else:
        
        logger.info(mess=f'No precomputation found at {words_precomputed_fp}. Loading WordNet from scratch')
        
        wordnet = WordNet(
            wordnet_fp=words_fp, 
            hierarchy_fp=hierarchy_fp,
            logger=logger
        )

        # Dump precomputed words for future use
        wordnet.dump_words(fp=WORDNET_DIR)
    
    return wordnet

def load_imagenet(wordnet: WordNet | None = None, logger: Logger = SilentLogger()) -> Tuple[ImageNetWords, ImageNetWords]:
    
    wordnet = default(wordnet, load_wordnet(logger))
    
    # Load ImageNet
    logger.info(mess='Loading ImageNet')
    inet_fp = os.path.join(WORDNET_DIR, 'imagenet_class_index.json')
    inet = ImageNetWords(imagenet_fp=inet_fp, wordnet=wordnet)
    
    # Load SuperImageNet
    logger.info(mess='Loading ImageNet superclasses')
    inet_super_fp = os.path.join(WORDNET_DIR, 'imagenet_superclass_index.json')
    inet_super = ImageNetWords(imagenet_fp=inet_super_fp, wordnet=wordnet)
    
    return inet, inet_super
    


# --- PLOTTING ---

def boxplots(
    data      : Dict[str, List[float]],
    ylabel    : str,
    title     : str,
    file_name : str,
    out_dir   : str,
    logger    : Logger = SilentLogger()
):
    '''
    Plot data in a boxplot and save it in the output directory.

    :param data: Data to plot indexed by cluster type.
    :type data: Dict[str, List[float]]
    :param ylabel: Label for the y-axis.
    :type ylabel: str
    :param title: Title of the plot.
    :type title: str
    :param file_name: Name of the file to save the plot.
    :type file_name: str
    :param out_dir: Output directory.
    :type out_dir: str
    :param logger: Logger to log the process, defaults to SilentLogger().
    :type logger: Logger, optional
    '''
    
    # --- MACROS ---
    FIGSIZE      = (12, 6)
    PALETTE      = [COLORS[CLU_ORDER[clu_name]] for clu_name in data]
    
    # Font customization
    FONT = 'serif'
    LABEL_ARGS   = {'fontsize': 16, 'labelpad': 10, 'fontfamily': FONT}
    TITLE_ARGS   = {'fontsize': 17, 'fontfamily': FONT}
    
    X_TICK_ARGS    = {'size': 12, 'rotation': 0, 'family': FONT}
    GRID_ARGS    = {'linestyle': '--', 'alpha': 0.7}

    # Create the plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    data_values = list(data.values())
    
    # Boxplot
    sns.boxplot(data=data_values, ax=ax, palette=PALETTE)

    # Set labels, title, and customize ticks
    ax.set_title(title, **TITLE_ARGS,)
    ax.set_ylabel(ylabel, **LABEL_ARGS)
    ax.set_xticks(np.arange(len(data_values)))
    ax.set_xticklabels(data.keys(), **X_TICK_ARGS)
    ax.set_xticklabels(data.keys(), **X_TICK_ARGS)

    # Add grid with custom settings
    ax.grid(True, **GRID_ARGS)

    # Ensure a tight layout
    fig.tight_layout()

    # Save the plot
    out_fp = os.path.join(out_dir, f'{file_name}.svg')
    logger.info(f'Saving plot to {out_fp}')
    fig.savefig(out_fp, bbox_inches='tight')



# --- RUN ---

class CurveFitter:

    def __init__(
        self,
        function: Callable[[float, ...], float],  # type: ignore
        popt: List[float],
        domain: Tuple[float, float] | None
    ):
        self.function = function
        self.popt     = popt
        self.domain   = domain

    @staticmethod
    def hyperbolic_quadratic(
        x: float,
        a: float, b: float, c: float, # Hyperbolic part 
        d: float, e: float            # Quadratic part
    ) -> float:

        # $$ a ** x  $$

        return  (a * x**2 + b * x + c) / (d * x + e)


    def __call__(self, x: float) -> float:

        if self.domain is not None:
            if x < self.domain[0] or x > self.domain[1]:
                raise ValueError(f'x={x} is out of domain {self.domain}')

        return self.function(x, *self.popt)