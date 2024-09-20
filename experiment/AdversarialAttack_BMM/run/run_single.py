'''
TODO Experiment description
'''

import matplotlib

from experiment.AdversarialAttack_BMM.args import ARGS
from experiment.utils.misc import run_single
from experiment.AdversarialAttack_BMM.adversarial_attack_max import AdversarialAttackMaxExperiment2

matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=AdversarialAttackMaxExperiment2)y