'''
TODO Experiment description
'''

from experiment.AdversarialAttack_BMM.args import ARGS
from experiment.utils.misc import run_multi
from experiment.AdversarialAttack_BMM.adversarial_attack_max import BMMMultiExperiment, AdversarialAttackMaxExperiment2

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=AdversarialAttackMaxExperiment2,
    multi_exp_type=BMMMultiExperiment
)
