import os

import numpy as np
from   numpy.typing import NDArray
import pandas as pd
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from analysis.utils.settings import CLUSTER_DIR, FILE_NAMES, LAYER_SETTINGS, OUT_DIR, OUT_NAMES, WORDNET_DIR
from analysis.utils.misc import end, start
from zdream.utils.logger import Logger, LoguruLogger, SilentLogger
from zdream.utils.io_ import read_json
from sklearn.decomposition import PCA

# ------------------------------------------- SETTINGS ---------------------------------------

LAYER    = 'fc8'

K        =        3 # Number of points to skip for the text
FIGSIZE  = (10, 10)
FONTSIZE =        6

CLU_DIR, NAME, TRUE_CLASSES, _ = LAYER_SETTINGS[LAYER]

OUT_NAME = f'{OUT_NAMES["cluster_type_comparison"]}_{NAME}'
cluster_dir = os.path.join(CLUSTER_DIR, CLU_DIR)

# ---------------------------------------------------------------------------------------------

def plot_embedding(
    df             : pd.DataFrame,
    label_col      : str | None = None, 
    label_txt      : str | None = None,
    out_fp         : str        = '.',
    title          : str        = 'Embeddings',
    fn             : str        = 'embedding',
    repeat_label   : bool       = True,
    logger         : Logger     = SilentLogger(),
):
    '''
    Plot t-SNE embedding

    :param df: Dataframe with TSNE embedding, numeric cluster labelings and classes associated to each point
    :type df: pd.DataFrame
    :param label_col: Label column cor the hue, defaults to None indicating no hue
    :type label_col: str | None, optional
    :param label_txt: Label column for the text, defaults to None indicating no text
    :type label_txt: str | None, optional
    :param out_fp: Output directory, defaults to current directory
    :type out_fp: str, optional
    :param title: Plot title
    :type title: str, optional
    :param fn: Output filename
    :type fn: str, optional
    :param repeat_label: Repeat label on plot, defaults to True
    :type repeat_label: bool, optional
    '''
    
    # Use as many colors as clusters
    palette = sns.color_palette(
        cc.glasbey, n_colors=df[label_col].nunique()  # type: ignore
    )  if label_col else None 
    
    for embedding_type in ['tsne', 'pca']:
    
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        x_name = f'Dim1-{embedding_type}'
        y_name = f'Dim2-{embedding_type}'
        
        # Plot the t-SNE embedding with the hue as input label
        sns.scatterplot(
            data=df, 
            x=x_name,
            y=y_name,
            hue=label_col, 
            palette=palette,
            ax=ax
        )
        
        # Add text to each point every k points
        if label_txt is not None:
            
            # Keep track of the labels already plotted
            labels =[]
            
            K = 1
            
            for i, txt in enumerate(df[label_txt]):
                
                # Check if tp plot depending on the two strategies
                if (repeat_label and i % K != 0) or (not repeat_label and txt in labels): 
                    continue
                
                # Add the label to the list and plot
                labels.append(txt)
                ax.annotate(txt, (df[x_name][i], df[y_name][i]), fontsize=FONTSIZE)
        
        # Add title and labels
        ax.set_xlabel(f'{embedding_type.capitalize()} Dimension 1')
        ax.set_ylabel(f'{embedding_type.capitalize()} Dimension 2')
        ax.set_title (f'{embedding_type.capitalize()} Embedding - {title}')
        plt.legend().remove()
        
        # Save 
        fig_fp = os.path.join(out_fp, f'{fn}-{embedding_type}.svg')
        logger.info(mess=f'Saving plot to {fig_fp}')
        fig.savefig(fig_fp)

if __name__ == '__main__':

    # 0. Creating logger and output directory

    logger = LoguruLogger(on_file=False)

    out_dir = os.path.join(OUT_DIR, OUT_NAME)
    logger.info(mess=f'Creating analysis target directory to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(mess=f'')

    # 1. LOADING LABELINGS and CLASSES

    start(logger, 'Loading data structures')

    labelings_fp = os.path.join(out_dir, FILE_NAMES['labelings'])
    logger.info(mess=f'Loading labelings from {labelings_fp}')
    labelings = dict(np.load(labelings_fp, allow_pickle=True))

    classes_fp      = os.path.join(WORDNET_DIR, FILE_NAMES['imagenet'])
    superclasses_fp = os.path.join(WORDNET_DIR, FILE_NAMES['imagenet_super'])

    logger.info(mess=f'Loading classes from {classes_fp}')
    classes = [label for _, label in read_json(classes_fp).values()]

    logger.info(mess=f'Loading superclasses from {superclasses_fp}')
    superclass = [label for _, label in read_json(superclasses_fp).values()]

    end(logger)

    # 2. TSNE Embedding

    start(logger, 't-SNE Embedding')

    recordings_fp = os.path.join(cluster_dir, FILE_NAMES['recordings'])
    logger.info(mess=f'Loading recordings from {recordings_fp}')
    recordings = np.load(recordings_fp)

    logger.info(mess='Computing t-SNE embedding up to 2 dimensions')
    tsne = TSNE(n_components=2)
    recordings_tsne = tsne.fit_transform(recordings)

    end(logger)
    
    # 3. PCA Embedding

    start(logger, 'PCA Embedding')

    logger.info(mess='Computing PCA embedding up to 2 dimensions')
    pca = PCA(n_components=2)
    recordings_pca = pca.fit_transform(recordings)

    end(logger)

    # 4. PLOTTING

    start(logger, 'Plotting embeddings')

    df_dict = {
        f'Dim1-tsne' : recordings_tsne[:, 0], 
        f'Dim2-tsne' : recordings_tsne[:, 1],
        f'Dim1-pca'  : recordings_pca [:, 0],
        f'Dim2-pca'  : recordings_pca [:, 1],
    }

    if TRUE_CLASSES:
        
        df_dict.update({
            'Class'     : np.array(classes),
            'Superclass': np.array(superclass)
        })    

    for k, v in labelings.items(): df_dict[k] = v

    df = pd.DataFrame(df_dict)

    embedding_dir = os.path.join(out_dir, 'embeddings')
    logger.info(mess=f'Creating embeddings directory to {embedding_dir}')
    os.makedirs(embedding_dir, exist_ok=True)

    if TRUE_CLASSES:
        
        logger.info(mess='Plotting embeddings with class labels')
        plot_embedding(
            df=df, 
            logger=logger, 
            label_txt='Class', out_fp=embedding_dir, title='Classes', fn='Classes')

        logger.info(mess='Plotting superclass embeddings')
        plot_embedding(
            df=df, logger=logger, 
            label_col='True', 
            label_txt='Superclass', 
            out_fp=embedding_dir,
            title='Superclasses', 
            fn='Superclasses', 
            repeat_label=False
        )

    for k in labelings.keys():
        
        logger.info(mess=f'Plotting {k} embeddings')
        plot_embedding(
            df=df, 
            logger=logger, 
            label_col=k, 
            out_fp=embedding_dir, 
            title=k, 
            fn=k
        )

    end(logger)

    logger.close()