
import os
from os import path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import colorcet as cc

from analysis.utils.misc import load_clusters, load_imagenet
from analysis.utils.settings import CLUSTER_DIR, LAYER_SETTINGS, OUT_DIR
from experiment.utils.misc import make_dir
from zdream.utils.io_ import load_pickle, store_pickle
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger

# --- SETTINGS ---

LAYER = 'conv5-maxpool'
K           =        4 # Number of points to skip for the text
MAX_REPEAT  =        3

FIGSIZE    = (10, 10)
FONTSIZE   = 5
POINT_SIZE = 40

TSNE_PERP  = [2, 5, 10, 15, 20, 30, 50, 100]
TSNE_STEPS = 5000


# --- FUNCTIONS ---

def plot_embedding(
    df           : pd.DataFrame,
    column       : str | None = None, 
    labels_col   : str | None = None,
    out_fp       : str        = '.',
    repeat_label : bool       = False,
    title        : str        = 'Embeddings',
    file_name    : str        = 'embedding',
    logger       : Logger     = SilentLogger(),
):
    
    # Use as many colors as clusters
    palette = sns.color_palette(
        cc.glasbey, n_colors=df[column].nunique()  # type: ignore
    )  if column else None 
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    x_name = f'Dim1'
    y_name = f'Dim2'
    
    # Plot the t-SNE embedding with the hue as input label
    sns.scatterplot(
        data=df, 
        x=x_name,
        y=y_name,
        hue=column, 
        palette=palette,
        ax=ax,
        s=POINT_SIZE
    )
    
    # Add text to each point every k points
    if labels_col is not None:
        
        seen_labels = []
        
        for i, txt in enumerate(df[labels_col]):
            
            if (repeat_label and i % K == 0) or (not repeat_label and seen_labels.count(txt) < MAX_REPEAT):
            
                ax.annotate(txt, (df[x_name][i], df[y_name][i]), fontsize=FONTSIZE)
                seen_labels.append(txt)
    
    # Add title and labels
    ax.set_xlabel(f'TSNE Dimension 1')
    ax.set_ylabel(f'TSNE Dimension 2')
    ax.set_title (f'TSNE Embedding - {title}')
    plt.legend().remove()
    
    # Save 
    fig_fp = os.path.join(out_fp, f'{file_name}.svg')
    logger.info(mess=f'Saving plot to {fig_fp}')
    fig.savefig(fig_fp)
    
    plt.close()


# --- RUN ---


def main():
    
    # 1. INITIALIZATION
    
    out_dir = os.path.join(OUT_DIR, "clustering_analysis")
    clu_dir = os.path.join(CLUSTER_DIR, LAYER_SETTINGS[LAYER]['directory'])

    logger = LoguruLogger(on_file=False)
    clusters = load_clusters(dir=clu_dir, logger=logger)
    embeddings_dir = make_dir(os.path.join(out_dir, 'embeddings', LAYER))
    logger.info("")
    
    embeddings_fp = path.join(embeddings_dir, 'embeddings.pkl')
    if path.exists(embeddings_fp):
        logger.info(f'Loading precomputed embeddings {embeddings_fp}')
        embeddings = load_pickle(embeddings_fp)
    else:
        embeddings = {}

    
    imagenet_class, imagenet_superclass = load_imagenet(logger=logger)
    
    classes    = [word.name for word in imagenet_class     ] # type: ignore
    superclass = [word.name for word in imagenet_superclass] # type: ignore
    
    # 2. TSNE Embedding
    
    recordings_fp = os.path.join(clu_dir, 'recordings.npy')
    logger.info(mess=f'Loading recordings from {recordings_fp}')
    recordings = np.load(recordings_fp)
    
    for perp in TSNE_PERP:

        embeddings_dir_perp = make_dir(os.path.join(embeddings_dir, f'perplexity_{perp}'))
        
        logger.info(mess=f'Computing t-SNE embedding up to 2 dimensions with perplexity={perp}')
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=TSNE_STEPS)
        
        perp_s = str(perp)
        if perp_s in embeddings: 
            recordings_tsne = embeddings[perp_s]
        else:             
            recordings_tsne = tsne.fit_transform(recordings)
            embeddings[perp_s] = recordings_tsne

        
        # 3. CREATING DATAFRAME
        df_dict = {
            'Dim1'       : recordings_tsne[:, 0], 
            'Dim2'       : recordings_tsne[:, 1],
            'Points'     : np.ones(recordings_tsne.shape[0]),
        }
        
        if LAYER_SETTINGS[LAYER]['has_labels']:
            
            df_dict.update({
                'Class'      : np.array(classes),
                'Superclass' : np.array(superclass),
            })
        
        for name, cluster in clusters.items():
            df_dict[name] = cluster.labeling
            
        df = pd.DataFrame(df_dict)
        
        plot_embedding(
            df=df,
            logger=logger,
            column="Points",
            out_fp=embeddings_dir_perp,
            title=f'Points - Perp: {perp}',
            file_name='points'
        )
            
        for cluster_name, cluster in clusters.items():
            
            label_txt    = ('Class' if cluster_name != 'True' else 'Superclass') if LAYER_SETTINGS[LAYER]['has_labels'] else None
            repeat_label = (cluster_name != 'True') if LAYER_SETTINGS[LAYER]['has_labels'] else False
            
            plot_embedding(
                df=df,
                logger=logger,
                column=cluster_name,
                repeat_label=repeat_label,
                labels_col=label_txt,
                out_fp=embeddings_dir_perp,
                title=f'{cluster_name} - Perp: {perp}',
                file_name=cluster_name
            )
    
    logger.info(f'Storing precomputed embeddings to {embeddings_fp}')
    store_pickle(embeddings, embeddings_fp)
    
    logger.info(mess='')
    logger.close()
    
if __name__ == '__main__': 
    
    #for layer in LAYER_SETTINGS:
    #    
    #    
    #    LAYER = layer
    #
    #    out_dir = os.path.join(OUT_DIR, "clustering_analysis")
    #    clu_dir = os.path.join(CLUSTER_DIR, LAYER_SETTINGS[LAYER]['directory'])
    #    
    #    EMBEDDINGS_FP = path.join(out_dir, 'embeddings.pkl')
    #    EMBEDDINGS    = load_pickle(EMBEDDINGS_FP) if path.exists(EMBEDDINGS_FP) else {}
    #    
    #    main()
    
    main()
