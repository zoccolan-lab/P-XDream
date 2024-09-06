import math
from io import BytesIO
from os import path
from typing import List, Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as ImgType
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.transforms.functional import to_pil_image

from analysis.utils.misc import AlexNetLayerLoader
from analysis.utils.settings import ALEXNET_DIR, LAYER_SETTINGS, OUT_DIR
from experiment.utils.misc import make_dir
from zdream.generator import DeePSiMGenerator, Generator
from zdream.utils.dataset import MiniImageNet
from experiment.utils.args import WEIGHTS, DATASET
from analysis.script.feature_maps import COLORS, _squares_visualization
from zdream.utils.logger import LoguruLogger

LAYER             = 'conv5-maxpool'
FM_NUMBER         = 256
FM_SIZE           = 36
FM_SIDE           = int(math.sqrt(FM_SIZE))
CLU_NAME          = 'DominantSet'
GENERATOR_VARIANT = 'fc7'

def _add_title(img: ImgType, title: str) -> ImgType:
    
    FIGSIZE  = (5, 5)
    FONTSIZE = 15
    
    fig, ax = plt.subplots(figsize=FIGSIZE)  # Create figure
    ax.imshow(np.array(img))                 # Display image
    ax.set_title(title, fontsize=FONTSIZE)   # Set title
    ax.axis('off')                           # Hide axis

    # Save image to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    img = Image.open(buf)
    
    return img


def synthetic_image(code: NDArray, generator: Generator, title: str = "") -> ImgType:
    
    b_code = np.expand_dims(code, axis=0) # Batch code
    b_ten  = generator(b_code)            # Generate
    ten    = b_ten[0]                     # Unbatch tensor
    img    = to_pil_image(ten)            # Cast to image
    
    if title: img = _add_title(img, title)
    
    return img

def natural_image(idx: int, inet: MiniImageNet, title: str = "") -> ImgType:
    
    item = inet[idx]          # Select item
    ten  = item['images']     # Extract image tensor
    img  = to_pil_image(ten)  # Cast to image
    
    if title: img = _add_title(img, title)
    
    return img

def tuple_to_idx(tpl: Tuple[int, int, int]) -> int:
    
    fm_idx, i, j = tpl
    idx = fm_idx * FM_SIZE + i * FM_SIDE + j
    return idx

def best_natural(seg_info, recordings: NDArray, inet: MiniImageNet, fm_idx: str, clu_idx: str, title: str = "") -> ImgType:
    
    idx = seg_info[str(fm_idx)][clu_idx]
    idx_ = [tuple_to_idx(key) for key in idx]
    
    optim_fitness = recordings[idx_, :].mean(axis=0)
    best_idx      = int(np.argmax(optim_fitness))
    best_fit      = float(optim_fitness[best_idx])
    
    return natural_image(idx=best_idx, inet=inet, title=f'{title} - Fitness: {best_fit:.3f}')

def get_squares(seg_info, name, fm_idx: str, clu_idx: str) -> ImgType:
    
    idx = seg_info[str(fm_idx)][clu_idx]
    idx_ = [tuple_to_idx(key) % FM_SIZE for key in idx]
    
    matrix = np.zeros(FM_SIZE, dtype=int)
    matrix[idx_] = 1

    # rowor map
    colors = COLORS 
    n_row = len(COLORS)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(n_row + 1) - 0.5, ncolors=n_row)
    
    if name == 'fm':
    
        title = f'Feature Map {fm_idx}'
        if clu_idx != 'all': 
            title += f' - Cluster {clu_idx}'
            
    elif name == 'clu':
        
        title = f'Cluster {fm_idx}'
        if clu_idx != 'all': 
            title += f' - FeatureMap {clu_idx}'
        else:
            title += f' - All FeatureMaps'
    
    return _squares_visualization(matrix=matrix, title=title, cmap=cmap, norm=norm)

def create_pdf(image_dict: Dict[str, List[Tuple[ImgType, ImgType, ImgType]]], output_path: str = 'output.pdf') -> None:
    
    FIGSIZE = (11.7, 8.3)  # A4 landscape in inches
    DPI = 300  # Resolution

    # Create a PdfPages object to handle multi-page PDF export
    with PdfPages(output_path) as pdf:
        for key, image_tuples in image_dict.items():
            # Create a figure with subplots, each subplot for one image in the tuple
            fig, axes = plt.subplots(3, len(image_tuples), figsize=FIGSIZE, dpi=DPI)
            
            if len(image_tuples) == 1:
                axes = np.array([axes]).T  # Ensure axes is a 2D array even if there's only one column
                
            for col, triple in enumerate(image_tuples):
                for row, img in enumerate(triple):
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')
            
            # Adjust spacing to minimize gaps between images
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
            
            # Save the current figure to the PDF
            pdf.savefig(fig)
            
            # Close the current figure to free memory
            plt.close(fig)


def main():

    logger = LoguruLogger(on_file=False)

    dataset   = MiniImageNet(root=DATASET)
    generator = DeePSiMGenerator(root=WEIGHTS, variant=GENERATOR_VARIANT)
    loader    = AlexNetLayerLoader(alexnet_dir=ALEXNET_DIR, layer=LAYER, logger=logger)

    recordings          = loader.load_recordings()
    fm_seg,   clu_seg   = loader.load_segmentation_info()
    fm_super, clu_super = loader.load_segmentation_superstimuli()


    for name, seg, superstim in zip(['fm', 'clu'], [fm_seg, clu_seg], [fm_super, clu_super]):
        
        out_dir = make_dir(path.join(OUT_DIR, 'feature_maps', LAYER, f'{name}_segmentation', 'superstimuli'))

        for clu_algo in superstim.keys():

            seg_ = seg[clu_algo]
            superstim_ = superstim[clu_algo]

            fm_pages: Dict[str, List[Tuple[ImgType, ImgType, ImgType]]] = {}

            for idx, cluster_info in superstim_.items():

                images: List[Tuple[ImgType, ImgType, ImgType]] = []

                for clu_idx, info in cluster_info.items():

                    if   name == 'fm' : logger.info(f'Feature Map {idx} - Cluster {clu_idx}')
                    elif name == 'clu': logger.info(f'Cluster {idx} - Feature Map {clu_idx}')
                    
                    tpl_images : Tuple[ImgType, ImgType, ImgType] = (
                        get_squares    (seg_info=seg_, name=name, fm_idx=idx, clu_idx=clu_idx),
                        best_natural   (seg_info=seg_, recordings=recordings, inet=dataset, fm_idx=idx, clu_idx=clu_idx, title='Best Natural Image'),
                        synthetic_image(code=info['code'], generator=generator, title=f'Best Synthetic Image - Fitness: {info["fitness"]:.3f}')
                    )
                    
                    images.append(tpl_images)

                fm_pages[idx] = images

        create_pdf(fm_pages, path.join(out_dir, f'{clu_algo}.pdf'))

if __name__ == '__main__': main()