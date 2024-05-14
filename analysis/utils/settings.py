
import os

from zdream.utils.io_ import read_json

# --- LOCAL DIRECTORIES ---

SETTINGS_FILE = os.path.abspath(os.path.join(__file__, '..', '..', 'local_settings.json'))
settings = read_json(SETTINGS_FILE)

OUT_DIR     = settings['out_dir']
WORDNET_DIR = settings['wordnet_dir']
CLUSTER_DIR = settings['cluster_dir']
WORDNET_DIR = settings['wordnet_dir']

# --- FILE NAMES ---

FILE_NAMES = {
    'words'            : 'words.txt',                       # WordNet dir
    'hierarchy'        : 'wordnet.is_a.txt',                # WordNet dir
    'imagenet'         : 'imagenet_class_index.json',       # WordNet dir
    'imagenet_super'   : 'imagenet_superclass_index.json',  # WordNet dir
    'words_precoputed' : 'words.pkl',                       # WordNet dir
    'labelings'        : 'labelings.npz',                   # Output  dir
    'recordings'       : 'recordings.npy',                  # Cluster dir          
    'affinity_matrix'  : 'affinity_matrix.npy',             # Cluster dir
    'ds_clusters'      : 'DSClusters.json',                 # Cluster dir
    'gmm_clusters'     : 'GMMClusters.json',                # Cluster dir
    'nc_clusters'      : 'NCClusters.json',                 # Cluster dir
    'fm_clusters'      : 'FeatureMap.json',                 # Cluster dir
}

# --- LAYER SETTINGS ---

LAYER_SETTINGS = {
    # Layer-name    : (directory,       format name,           use ground truth, number of clusters, use feature map)
    'fc8'           : ('fc8',           'alexnetfc8',          True,             50,                 False),
    'fc7-relu'      : ('fc7-relu',      'alexnetfc7relu',      False,            127,                False),
    'fc7'           : ('fc7',           'alexnetfc7',          False,            58,                 False),
    'fc6-relu'      : ('fc6-relu',      'alexnetfc6relu',      False,            94,                 False),
    'conv5-maxpool' : ('conv5-maxpool', 'alexnetconv5maxpool', False,            576,                True ),
}

LAYER = 'conv5-maxpool'


# --- OUT NAMES ---

OUT_NAMES = {
    'cluster_type_comparison': 'clutype_comparison'
}
