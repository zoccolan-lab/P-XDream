import io
import math
import os
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from  PIL.Image import Image as ImgType

from zdream.generator import DeePSiMGenerator, Generator
from zdream.clustering.cluster import Cluster
from zdream.clustering.ds import DSCluster
from zdream.utils.dataset import MiniImageNet
from zdream.utils.logger import LoguruLogger
from analysis.utils.misc import AlexNetLayerLoader
from analysis.utils.settings import ALEXNET_DIR, LAYER_SETTINGS, OUT_DIR
from experiment.utils.args import DATASET, WEIGHTS
from experiment.utils.misc import make_dir

LAYER = 'fc8'
CLU_GEN_VARIANT = 'fc7'
UNITS_GEN_VARIANT = 'fc8'

# SYNTHETIC AND NATURAL TO IMAGES

def synthetic_image(generator: Generator, code: NDArray) -> ImgType:
    
    b_code = np.expand_dims(code, axis=0) # Batch code
    b_ten  = generator(b_code)            # Generate
    ten    = b_ten[0]                     # Unbatch tensor
    img    = to_pil_image(ten)            # Cast to image
    
    return img

def natural_image(inet: MiniImageNet, idx: int) -> ImgType:
    
    item = inet[idx]          # Select item
    ten  = item['images']     # Extract image tensor
    img  = to_pil_image(ten)  # Cast to image
    
    return img

# BEST ACTIVATION FROM RECORDINGS

def best_natural_superstimulus(cluster: Cluster, recordings: NDArray, weighted: bool = False) -> Tuple[int, float]:

    if not isinstance(cluster, DSCluster) and weighted: raise ValueError('Weighted clustering only supported for DSCluster')
    
    if weighted: weights = cluster.ranks                          # Weighted   # type: ignore - convered by error
    else:        weights =  np.ones(len(cluster)) / len(cluster)  # Arithmetic
    
    cluster_units = recordings[cluster.labels, :] # type: ignore
    cluster_map   = cluster_units * weights[:, np.newaxis]
    average       = np.sum(cluster_map, axis=0) 
    max_arg       = np.argmax(average)
    
    return int(max_arg), average[max_arg]

def best_natural_unit(label: int, recordings: NDArray) -> Tuple[int, float]:
    
    unit    = recordings[label, :]
    max_arg = np.argmax(unit)
    
    return int(max_arg), float(unit[max_arg])

# RENDERING

def superstimulus_grid(superstimuli: List[Tuple[Image, str]], main_title: str = "", **kwargs) -> ImgType: # type: ignore
    
    FIGSIZE = (10, 10)
    NROW    = math.ceil(math.sqrt(len(superstimuli)))
    NCOL    = len(superstimuli) // NROW
    
    if NROW * NCOL != len(superstimuli): NCOL += 1
    
    FONTDICT = {
        'fontsize'   : kwargs.get('fontsize',          7),
        'fontweight' : kwargs.get('fontweight', 'medium'),
        'family'     : kwargs.get('family',      'serif')
    }
    
    TITLE_FTSIZE = kwargs.get('title_fontsize', 24)
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(NROW, NCOL, figsize=FIGSIZE)
    
    if len(superstimuli) > 1: axes = axes.flatten()
    else                    : axes = [axes]

    for i, (img, title) in enumerate(superstimuli):
        axes[i].imshow(img)      # Display the image
        axes[i].set_title(title, fontdict=FONTDICT) # Add the title
        axes[i].axis('off')      # Remove the axis
    
    i += 1
    while i < NROW * NCOL:
        axes[i].axis('off')  # Hide the axis
        axes[i].imshow(Image.new('RGB', (1, 1), color='white'))  # Optionally add a white image
        i += 1
        
    fig.suptitle(main_title, fontsize=TITLE_FTSIZE, fontweight=FONTDICT['fontweight'], family=FONTDICT['family'])
    
    plt.tight_layout(pad=2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
    # Save the plot to a BytesIO object in PNG format
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the plot to free memory
    plt.close(fig)
    
    # Convert the buffer to a PIL image and return it
    img = Image.open(buf)
        
    return img

def images_to_pdf(image_triples, pdf_path):
    # Use the size of a horizontal A4 page (landscape orientation)
    PAGE_WIDTH_INCHES = 16  # A4 width in inches (landscape)
    PAGE_HEIGHT_INCHES = 6  # A4 height in inches (landscape)
    DPI = 500  # High resolution DPI
    
    # Create a PdfPages object to handle multi-page PDF export
    with PdfPages(pdf_path) as pdf:
        for triple in image_triples:
            fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES), dpi=DPI)
            
            for ax, img in zip(axes, triple):
                ax.imshow(img)
                ax.axis('off')
            
            # Adjust spacing to fill the page
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)
            
            # Save the current figure to the PDF
            pdf.savefig(fig)
            
            # Close the current figure to free memory
            plt.close(fig)

def main():

    logger    = LoguruLogger(on_file=False)
    out_dir   = make_dir(os.path.join(OUT_DIR, "clustering_analysis", "visualization", LAYER_SETTINGS[LAYER]['directory']), logger=logger)
    
    alexnet_loader    = AlexNetLayerLoader(alexnet_dir=ALEXNET_DIR, layer=LAYER, logger=logger)

    clusters = alexnet_loader.load_clusters()
    clu_superstimuli, units_supertimuli = alexnet_loader.load_superstimuli()
    recordings = alexnet_loader.load_recordings()

    logger.info(f'Loading ImageNet from {DATASET}')
    inet              = MiniImageNet(DATASET)

    logger.info('Loading generator variants')
    generator_clu     = DeePSiMGenerator(root=WEIGHTS, variant=CLU_GEN_VARIANT)
    generator_units   = DeePSiMGenerator(root=WEIGHTS, variant=UNITS_GEN_VARIANT)

    for clu_algo in clusters:

        logger.info(f'Processing {clu_algo} clusters')

        clu_images = []

        for clu_idx, cluster in enumerate(clusters[clu_algo]):  # type: ignore
            
            ROUND = 2
            
            images     = []
            images_syn = []
            images_nat = []

            if clu_algo == 'DominantSet':

                for weighted in [False, True]:

                    superstimuli = clu_superstimuli[f'{clu_algo}{"Weighted" if weighted else ""}'][clu_idx]

                    code        = superstimuli['code']
                    syn_fitness = np.max(superstimuli['fitness'])
                    
                    synthetic_img = synthetic_image(code=code, generator=generator_clu)
                    
                    nat_idx, nat_fitness = best_natural_superstimulus(cluster=cluster, weighted=weighted, recordings=recordings)
                    natural_img = natural_image(idx=nat_idx, inet=inet)

                    weight_name = "Weighted" if weighted else "Arithmetic"
                    
                    images.append((synthetic_img, f"Best syntethic - {weight_name} average - Fitness: {round(syn_fitness, ROUND)}"))
                    images.append((  natural_img, f"Best natural   - {weight_name} average - Fitness: {round(nat_fitness, ROUND)}"))
            
            else:

                superstimuli = clu_superstimuli[clu_algo][clu_idx]

                code = superstimuli['code']
                syn_fitness = np.max(superstimuli['fitness'])
                synthetic_img = synthetic_image(code=code, generator=generator_clu)

                nat_idx, nat_fitness = best_natural_superstimulus(cluster=cluster, recordings=recordings)
                natural_img = natural_image(idx=nat_idx, inet=inet)

                images.append((synthetic_img, f"Best syntethic - Fitness: {round(syn_fitness, ROUND)}"))
                images.append((  natural_img, f"Best natural   - Fitness: {round(nat_fitness, ROUND)}"))
            
            for label in cluster.labels:
                
                code        = units_supertimuli[label]['code']
                syn_fitness = units_supertimuli[label]['fitness']
                
                synthetic_img = synthetic_image(code=code, generator=generator_units)
                
                nat_idx, nat_fitness = best_natural_unit(label=label, recordings=recordings)
                natural_img = natural_image(idx=nat_idx, inet=inet)
                
                images_syn.append((synthetic_img, f"Unit {label} - Fitness: {round(syn_fitness, ROUND)}"))
                images_nat.append((  natural_img, f"Unit {label} - Fitness: {round(nat_fitness, ROUND)}"))
                
            clu_images.append((
                superstimulus_grid(
                    images, main_title=f"Cluster {clu_idx} - Collective fitness",
                    fontsize=11, title_fontsize=20
                ),
                superstimulus_grid(
                    images_syn, main_title=f"Cluster {clu_idx} - Units synthetic supertimuli", 
                    fontsize=7, title_fontsize=20
                ),
                superstimulus_grid(
                    images_nat, main_title=f"Cluster {clu_idx} - Units natural supertimuli",   
                    fontsize=7, title_fontsize=20
                )
            ))
        
        out_fp = os.path.join(out_dir, f'{clu_algo}.pdf')
        logger.info(f"Saving cluster visualization to {out_fp}")
        images_to_pdf(clu_images, out_fp)

if __name__ == "__main__":
    
    for layer in LAYER_SETTINGS:

        if layer == 'fc8': continue
        
        LAYER = layer
        main()