
from collections import defaultdict
import math
import os
import io
from typing import Dict, List, Tuple

import PIL
from PIL.Image import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score

from analysis.utils.misc import boxplots, load_clusters
from analysis.utils.settings import ALEXNET_DIR, LAYER_SETTINGS, OUT_DIR
from experiment.utils.misc import make_dir
from zdream.utils.io_ import save_json
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger

# --- SETTINGS ---

LAYER       = 'conv5-maxpool'
FM_SIZE     = 36
SIDE        = int(math.sqrt(FM_SIZE))  # Side of the feature map
FIGSIZE     = (5, 5)                   # Size of the plot
TITLE_FONT  = 18 #16                       # Font size of the title
COLOR_MAP   = plt.cm.viridis           # type: ignore - Color map clu_on_fm
COLORS      = np.array([
    [1.0, 1.0, 1.0, 1.0],  # White
    [0.2, 0.6, 1.0, 1.0],  # Black
])

# Pdf
FIGSIZE_PDF = (16, 6)                  # Size of the plot in the PDF
K           =   4                      # Number of images per page
DPI         = 300                      # PDF resolution

PLOTS = {
    'clu_on_fm'               : False,
    'clu_on_fm_visualization' : False,
    'clu_on_fm_optim_file'    : True,
    'fm_on_clu'               : False,
    'fm_on_clu_visualization' : False,
    'fm_on_clu_optim_file'    : True
}

# --- ROUTINES ---

def _u_to_tuple(idx: int) -> Tuple[int, int, int]:
    
    fm_idx = idx // FM_SIZE
    square = idx % FM_SIZE
    i = square // SIDE
    j = square % SIDE
    
    return int(fm_idx), int(i), int(j)

def _fm_visualization(matrix: NDArray, title: str = "", cmap=None, norm=None) -> Image:
    
    assert matrix.size == SIDE * SIDE, f'The number of values must be equal to {SIDE * SIDE} ({SIDE} * {SIDE})'
    
    matrix = matrix.reshape(SIDE, SIDE)
    
    # Create the plot using the colormap
    fig, ax = plt.subplots(figsize=FIGSIZE)
        
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', length=0)
    ax.set_title(title, fontsize=TITLE_FONT)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    # Save the plot to a BytesIO object in PNG format
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    img = PIL.Image.open(buf)  # type: ignore
    
    return img

def feature_map_visualization(
    values: List[Tuple[List[int], int]], 
    out_fp: str,
    logger: Logger = SilentLogger()
):
    
    images = []
    
    logger.info(mess=f'Processing feature maps...')
    
    # 1. Generate images
    
    for value, fm_number in values:
        
        # Create a colormap for the unique values
        unique_values = np.unique(value)    
        colors = COLOR_MAP(np.linspace(0, 1, len(unique_values)))
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_values) + 1) - 0.5, ncolors=len(unique_values))
        
        # Map matrix to color indexes
        mapping = {v: i for i, v in enumerate(unique_values)}
        matrix = np.array(value)
        matrix = np.vectorize(mapping.get)(matrix)
        
        # Generate image
        img = _fm_visualization(matrix=matrix, title=f'Feature map - {fm_number}', cmap=cmap, norm=norm)
        
        images.append(img)
        
    # 2. Generate images
    
    # Split the images into pages using K images per page
    one_page_images = [tuple(images[i:i+K]) for i in range(0, len(images), K)]
    
    # Create a PdfPages object to handle multi-page PDF export
    logger.info(mess=f'Saving pdf to {out_fp}...')
    
    with PdfPages(out_fp) as pdf:
        
        for imgs in one_page_images:
            
            fig, axes = plt.subplots(1, K, figsize=FIGSIZE_PDF, dpi=DPI)
            
            for ax, img in zip(axes, imgs):
                ax.imshow(img)
                ax.axis('off')
            
            plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.1)
            pdf.savefig(fig)
            plt.close(fig)

def cluster_visualization(
    values: Dict[int, Dict[int, List[int]]], 
    out_fp: str,
    logger: Logger = SilentLogger()
):
    
    def _cluster_visualization_aux(clu_idx: int, fm_number: int, fm_idx: List[int]):
        
        n_col = len(COLORS)

        assert 0 < len(fm_idx) <= FM_SIZE, f'Invalid number of feature maps: {len(fm_idx)}, expected 0 < len <= {FM_SIZE}'

        fm_clu = np.zeros(FM_SIZE, dtype=int)
        fm_clu[fm_idx] = 1

        # Color map
        colors = COLORS 
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(n_col + 1) - 0.5, ncolors=n_col)
        
        return _fm_visualization(matrix=fm_clu, title=f'Cluster {clu_idx} - Feature Map {fm_number}', cmap=cmap, norm=norm)
    
    # 1. Generate images
    
    images = []
    
    logger.info(mess=f'Processing feature maps...')
    
    for clu_idx, fm_info in values.items():
        
        images.append([
            _cluster_visualization_aux(clu_idx=clu_idx, fm_number=fm_number, fm_idx=fm_idx) 
            for fm_number, fm_idx in fm_info.items()
        ])
    
    # 2. Save as pdf
    blank_image = PIL.Image.new('RGB', (1, 1), 'white')  # type: ignore
    
    logger.info(mess=f'Saving pdf to {out_fp}...')
    
    with PdfPages(out_fp) as pdf:
        
        for imgs in images:
            
            if len(imgs) < K:
                fig, axes = plt.subplots(1, len(imgs), figsize=FIGSIZE_PDF, dpi=DPI)
            else:
                reminder = len(imgs) % K
                if reminder != 0: imgs += [blank_image] * (K - reminder)
                fig, axes = plt.subplots(len(imgs) // K, K, figsize=FIGSIZE_PDF, dpi=DPI)
            
            if len(imgs) == 1:
                axes.imshow(imgs[0])
                axes.axis('off')
            else:
                for ax, img in zip(axes.flatten(), imgs):
                    ax.imshow(img)
                    ax.axis('off')
            
            plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.1)
            pdf.savefig(fig)
            plt.close(fig)


# --- RUN ---


def main():

    out_dir = os.path.join(OUT_DIR, "feature_maps", LAYER_SETTINGS[LAYER]['directory'])
    clu_dir = os.path.join(ALEXNET_DIR, LAYER_SETTINGS[LAYER]['directory'])
    
    assert os.path.exists(os.path.join(clu_dir, 'FeatureMapClusters.json')), f'Feature map not found at : {clu_dir}'

    # Initialize logger
    logger = LoguruLogger(on_file=False)
    
    # Load clusters
    clusters = load_clusters(dir=clu_dir, logger=logger)
    
    #clusters['DominantSet']._clusters = clusters['DominantSet']._clusters[:10]
    #clusters = {'DominantSet': clusters['DominantSet']}
        
    # Cluster on feature map
    if any([PLOTS['clu_on_fm'], PLOTS['clu_on_fm_visualization'], PLOTS['clu_on_fm_optim_file']]):
        
        logger.info("Analyzing clusters mapping into feature maps...")
        
        logger.formatting = lambda x: f' > {x}'
        
        clu_on_fm_dir = make_dir(path=os.path.join(out_dir, 'clusters_on_feature_maps'), logger=logger)
        
        if PLOTS['clu_on_fm']            : fm_clu_count, fm_clu_entropy = {}, {}
        if PLOTS['clu_on_fm_optim_file'] : fm_clu_optim = {}
        
        for cluster_name, cluster in clusters.items():
            
            labeling = cluster.labeling
            
            clusters_on_featmaps = [
                (list(labeling[i:i+FM_SIZE]), fm_idx) 
                for fm_idx, i in enumerate(range(0, len(labeling), FM_SIZE))
            ]
            
            if PLOTS['clu_on_fm']:
                
                fm_clu_count  [cluster_name] = [len(np.unique(value))           for value, _ in clusters_on_featmaps]
                fm_clu_entropy[cluster_name] = [mutual_info_score(value, value) for value, _ in clusters_on_featmaps]
            
            if PLOTS['clu_on_fm_optim_file']:
                
                optimization  = {}

                for clu_info, fm_number in clusters_on_featmaps:
                    
                    optim = {}
                    
                    for unique in set(clu_info):
                        optim[int(unique)] = [_u_to_tuple(fm_number * 36 + i) for i, v in enumerate(clu_info) if v == unique]
                        
                    optim['all'] = [i for idx in optim.values() for i in idx]
                    
                    optimization[int(fm_number)] = optim
                
                fm_clu_optim[str(cluster_name)] = optimization
                
                
            if PLOTS['clu_on_fm_visualization']:
                
                vis_dir = make_dir(path=os.path.join(clu_on_fm_dir, 'visualization'), logger=logger)
                
                feature_map_visualization(
                    values=clusters_on_featmaps,
                    out_fp=os.path.join(vis_dir, f'{cluster_name}.pdf'),
                    logger=logger
                )
        
        if PLOTS['clu_on_fm']:
            
            boxplots(
                data=fm_clu_count,
                ylabel='Cardinality',
                title='Cluster Cardinality on Feature Maps',
                out_dir=clu_on_fm_dir,
                file_name='cluster_count',
                logger=logger
            )
            
            boxplots(
                data=fm_clu_entropy,
                ylabel='Cluster Entropy',
                title='Cluster Entropy on Feature Maps',
                out_dir=clu_on_fm_dir,
                file_name='cluster_entropy',
                logger=logger
            )
            
        if PLOTS['clu_on_fm_optim_file']:
            
            optim_file = os.path.join(clu_on_fm_dir, 'fm_segmentation_optim.json')
            
            logger.info(mess=f'Saving optimization files to {optim_file}')
            
            save_json(fm_clu_optim, optim_file)

    # Feature map on cluster
    if any([PLOTS['fm_on_clu'], PLOTS['fm_on_clu_visualization'], PLOTS['fm_on_clu_optim_file']]):
        
        logger.info("Analyzing feature maps mapping into clusters...")
        
        logger.formatting = lambda x: f' > {x}'
        
        fm_on_clu_dir = make_dir(path=os.path.join(out_dir, 'feature_maps_on_clusters'), logger=logger)
        
        if PLOTS['fm_on_clu']            : fm_clu_count, fm_clu_entropy = {}, {}
        if PLOTS['fm_on_clu_optim_file'] : fm_clu_optim = {}
        
        for cluster_name, clu in clusters.items():
            
            involved_fm: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

            for i, cluster in enumerate(clu): # type: ignore
                
                fm_involved = [(label // FM_SIZE, label % FM_SIZE) for label in cluster.labels]
                
                for fm_number, fm_idx in fm_involved:
                    involved_fm[i][fm_number].append(fm_idx)
            
            if PLOTS['fm_on_clu']:
                
                fm_clu_count[cluster_name] = [len(clu_info) for _, clu_info in involved_fm.items()]
                
                entropy_values = []
                for _, clu_info in involved_fm.items():
                    entropy_idx = []
                    for fm_number, fm_idx in clu_info.items():
                        entropy_idx.extend([fm_number] * len(fm_idx))
                    entropy_values.append(entropy_idx)
                fm_clu_entropy[cluster_name] = [mutual_info_score(value, value) for value in entropy_values]
            
            if PLOTS['fm_on_clu_optim_file']:
                
                optimization = {}

                for clu_idx, fm_info in involved_fm.items():
                    
                    optim: Dict[int | str, List] = {
                        int(fm_number): [_u_to_tuple(fm_number * FM_SIZE + idx) for idx in fm_idx]
                        for fm_number, fm_idx in fm_info.items()
                    }
                    
                    optim['all'] = [i for idx in optim.values() for i in idx]
                    
                    optimization[clu_idx] = optim
                
                fm_clu_optim[cluster_name] = optimization
            
            if PLOTS['fm_on_clu_visualization']:
                    
                vis_dir = make_dir(path=os.path.join(fm_on_clu_dir, 'visualization'), logger=logger)
                
                cluster_visualization(
                    values=involved_fm,
                    out_fp=os.path.join(vis_dir, f'{cluster_name}.pdf'),
                    logger=logger
                )
                
        if PLOTS['fm_on_clu']:
                
            boxplots(
                data=fm_clu_count,
                ylabel='Feature Map Cardinality',
                title='Feature Map Cardinality on Clusters',
                out_dir=fm_on_clu_dir,
                file_name='fm_count',
                logger=logger
            )
            
            boxplots(
                data=fm_clu_entropy,
                ylabel='Feature Map Entropy',
                title='Feature Map Entropy on Clusters',
                out_dir=fm_on_clu_dir,
                file_name='fm_entropy',
                logger=logger
            )
        
        if PLOTS['fm_on_clu_optim_file']:
            
            optim_file = os.path.join(fm_on_clu_dir, 'clu_segmentation_optim.json')
            
            logger.info(mess=f'Saving optimization files to {optim_file}')
            
            save_json(fm_clu_optim, optim_file)
            
        logger.reset_formatting()

if __name__ == '__main__': 
    
    main()