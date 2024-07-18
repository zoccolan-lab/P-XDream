import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from typing import Callable, Dict, List, Tuple
from analysis.utils.settings import CLU_ORDER, WORDNET_DIR
from analysis.utils.wordnet import ImageNetWords, WordNet
from zdream.clustering.cluster import Clusters
from zdream.utils.logger import Logger, SilentLogger
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

def load_clusters(dir: str, exclude: List[str] = [], logger: Logger = SilentLogger()) -> Dict[str, Clusters]:
    
    clusters = {}
    
    logger.info(mess=f'Loading clusters from {dir}')
    
    for root, _, files in os.walk(dir):
        
        for file in files:
            
            if file.endswith("Clusters.json") and file not in exclude:
                
                file_path = os.path.join(root, file)
                logger.info(mess=f' > Loading {file}')
                clusters[file.replace("Clusters.json", "")] = Clusters.from_file(fp=file_path)
                
            else:
                logger.info(mess=f' > Skipping {file}')
                
    clusters = dict(sorted(clusters.items(), key=lambda x: CLU_ORDER[x[0]]))

    return clusters

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

def box_violin_plot(
    data      : Dict[str, List[int]] | Dict[str, List[float]],
    ylabel    : str,
    title     : str,
    file_name : str,
    out_dir   : str,
    logger    : Logger = SilentLogger()
):
    '''
    Plot data in a boxplot and violinplot and save them in the output directory

    :param data: Data to plot indexed by cluster type.
    :type data: Dict[str, List[int]] | Dict[str, List[float]]
    :param ylabel: Label for the y axis.
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
    
    SNS_PALETTE = 'husl'
    FIGSIZE     = (10, 6)
    
    palette = sns.set_palette(SNS_PALETTE)
    
    for plot_ty in ['boxplot']: #, 'violinplot']:
    
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        data_values = list(data.values())
        
        if plot_ty == 'boxplot': sns.boxplot   (data=data_values, ax=ax, palette=palette)
        else:                    sns.violinplot(data=data_values, ax=ax, palette=palette)
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(data_values)))
        ax.set_xticklabels(data.keys())
        ax.tick_params(axis='x', rotation=45) 
        plt.tight_layout() 
        
        out_fp = os.path.join(out_dir, f'{file_name}_{plot_ty}.svg')
        logger.info(mess=f'Saving plot to {out_fp}')
        fig.savefig(out_fp)


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