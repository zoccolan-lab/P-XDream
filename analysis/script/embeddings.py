import os

import numpy as np
from   numpy.typing import NDArray
import pandas as pd
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from zdream.utils.logger import LoguruLogger
from zdream.utils.io_ import read_json

# ------------------------------------------- SETTINGS ---------------------------------------

SETTINGS_FILE = os.path.abspath(os.path.join(__file__, '..', '..', 'local_settings.json'))
settings = read_json(SETTINGS_FILE)

LAYER = 'fc7'

SETTINGS = {
    'fc8'     : ('fc8',      'alexnetfc8',     True),
    'fc7-relu': ('fc7-relu', 'alexnetfc7relu', False),
    'fc7'     : ('fc7',      'alexnetfc7',     False)
}

CLU_DIR, NAME, TRUE_CLASSES = SETTINGS[LAYER]

OUT_DIR     = settings['out_dir']
WORDNET_DIR = settings['wordnet_dir']
CLUSTER_DIR = os.path.join(settings['cluster_dir'], CLU_DIR)

FILE_NAMES = {
    'recordings'      : 'recordings.npy',
    'superclasses'    : 'imagenet_superclass_index.json',
    'classes'         : 'imagenet_class_index.json',
    'labelings'       : 'labelings.npz'
}

OUT_NAME = f'clusterings_labelings_{NAME}'  # the directory is supposed to contain the labelings

K        =        3 # Number of points to skip for the text
FIGSIZE  = (10, 10)
FONTSIZE =        6

# ---------------------------------------------------------------------------------------------

def start(logger, name):
    logger.info(mess=name)
    logger.formatting = lambda x: f'> {x}'

def done(logger):
    logger.info(mess='Done')
    logger.reset_formatting()
    logger.info(mess='')

def plot_tsne(
    df           : pd.DataFrame,
    logger       : LoguruLogger,
    label_col    : str | None = None, 
    label_txt    : str | None = None,
    out_fp       : str        = '.',
    title        : str        = 'Embeddings',
    fn           : str        = 'embedding',
    repeat_label : bool       = True
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
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Plot the t-SNE embedding with the hue as input label
    sns.scatterplot(
        data=df, 
        x='Dim1',
        y='Dim2',
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
            ax.annotate(txt, (df['Dim1'][i], df['Dim2'][i]), fontsize=FONTSIZE)
    
    # Add title and labels
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE Embedding - {title}')
    plt.legend().remove()
    
    # Save 
    fig_fp = os.path.join(out_fp, f'{fn}.svg')
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

    classes_fp      = os.path.join(WORDNET_DIR, FILE_NAMES['classes'])
    superclasses_fp = os.path.join(WORDNET_DIR, FILE_NAMES['superclasses'])

    logger.info(mess=f'Loading classes from {classes_fp}')
    classes = [label for _, label in read_json(classes_fp).values()]

    logger.info(mess=f'Loading superclasses from {superclasses_fp}')
    superclass = [label for _, label in read_json(superclasses_fp).values()]

    done(logger)

    # 2. TSNE Embedding

    start(logger, 't-SNE Embedding')

    recordings_fp = os.path.join(CLUSTER_DIR, FILE_NAMES['recordings'])
    logger.info(mess=f'Loading recordings from {recordings_fp}')
    recordings = np.load(recordings_fp)

    logger.info(mess='Computing t-SNE embedding up to 2 dimensions')
    tsne = TSNE(n_components=2)
    recordings_tsne = tsne.fit_transform(recordings)

    done(logger)

    # 3. PLOTTING

    start(logger, 'Plotting embeddings')

    df_dict = {
        'Dim1'      : recordings_tsne[:, 0], 
        'Dim2'      : recordings_tsne[:, 1],
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
        plot_tsne(
            df=df, 
            logger=logger, 
            label_txt='Class', out_fp=embedding_dir, title='Classes', fn='tsne_classes')

        logger.info(mess='Plotting superclass embeddings')
        plot_tsne(
            df=df, logger=logger, 
            label_col='True', 
            label_txt='Superclass', 
            out_fp=embedding_dir,
            title='Superclasses', 
            fn='tsne_superclasses', 
            repeat_label=False
        )

    for k in labelings.keys():
        
        logger.info(mess=f'Plotting {k} embeddings')
        plot_tsne(
            df=df, 
            logger=logger, 
            label_col=k, 
            out_fp=embedding_dir, 
            title=k, 
            fn=f'tsne_{k}'
        )

    done(logger)

    logger.close()