
import os
from typing import Dict

from pxdream.utils.io_ import read_json

# --- LOCAL DIRECTORIES ---

SETTINGS_FILE = os.path.abspath(os.path.join(__file__, '..', '..', 'local_settings.json'))
settings      = read_json(SETTINGS_FILE)

OUT_DIR             : str = settings['out_dir']
WORDNET_DIR         : str = settings['wordnet_dir']
#ALEXNET_DIR         : str = settings['alexnet_dir']
#NEURON_SCALING_FILE : str = settings['neuron_scaling_file']


# --- LAYER SETTINGS ---

LAYER_SETTINGS = {
    'fc8': {
        'directory'          : 'fc8',
        'format_name'        : 'alexnetfc8',
        'has_labels'         : True,
        'number_of_clusters' : 50,
        'feature_map'        : False,
        'neurons'            : 1000,
        'title'              : 'Layer Fc8',
    },
    'fc7-relu': {
        'directory'          : 'fc7-relu',
        'format_name'        : 'alexnetfc7relu',
        'has_labels'         : False,
        'number_of_clusters' : 127,
        'feature_map'        : False,
        'neurons'            : 4096,
        'title'              : 'Layer Fc7-ReLu',
    },
    'fc6-relu': {
        'directory'          : 'fc6-relu',
        'format_name'        : 'alexnetfc6relu',
        'has_labels'         : False,
        'number_of_clusters' : 94,
        'feature_map'        : False,
        'neurons'            : 4096,
        'title'              : 'Layer Fc6-ReLu',
    },
    'conv5-maxpool': {
        'directory'          : 'conv5-maxpool',
        'format_name'        : 'alexnetconv5maxpool',
        'has_labels'         : False,
        'number_of_clusters' : 576,
        'feature_map'        : True,
        'neurons'            : 9216,
        'title'              : 'Layer Conv5-MaxPool',
    },
}


# --- CLUSTER ORDER ---

COLORS = [
    "#bc3bee",
    "#59bb29",
    "#f74852",
    "#ffd82a",
    "#19c4b9",
    "#f78436",
    "#1f12d5",
    "#797171",
]

CLU_ORDER = {
    'DominantSet'         : 0,
    'DominantSetWeighted' : 0.1,
    'NormalizedCut'       : 1,
    'GaussianMixture'     : 2,
    'DBSCAN'              : 3,
    'Adjacent'            : 4,
    'Random'              : 5,
    'FeatureMap'          : 6,
    'Semantic'            : 7,
}