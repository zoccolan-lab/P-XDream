

# %%
import numpy as np
from tqdm import trange
import torch
from torch import Tensor
from PIL import Image
from einops import rearrange
from torchvision.models import list_models
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from typing import cast, Tuple
from collections import defaultdict

from zdream.generator import InverseAlexGenerator
from zdream.probe import RecordingProbe
from zdream.subject import NetworkSubject
from zdream.utils import MiniImageNet, device, repeat_pattern, logicwise_function
from zdream.scores import MaxActivityScorer
from zdream.optimizer import GeneticOptimizer

population_size = 20


#instantiate generator with appropriate transformation and mini imagenet loader
def transform(
    imgs : Tensor,
    mean : Tuple[int, ...] = (104.0, 117.0, 123.0), # type: ignore
    raw_scale : float = 255.
) -> Tensor:
    mean : Tensor = torch.tensor(mean, device=imgs.device).reshape(-1, 1, 1)

    imgs += mean
    imgs /= raw_scale

    return imgs.clamp(0, 1)

mini_IN = MiniImageNet(root="/home/lorenzo/Desktop/Datafolders/tiny imagenet")
mini_IN_loader = DataLoader(mini_IN, batch_size=2, shuffle=True)
mask = repeat_pattern(n = population_size, base_seq = [True, False, False, False, False,False,False], rand = False)

gen = InverseAlexGenerator(
        root='/home/lorenzo/Desktop/Datafolders/ZXDREAM/Kreiman_Generators',
        variant='fc8',
        output_pipe=transform,
        nat_img_loader = mini_IN_loader
).to(device)


#initialize the network subject (alexnet) with a recording probe
l = '20_linear_03' #layer you want to record from
record_dict = {l: None} #i.e. fc8
my_probe = RecordingProbe(target = record_dict)
sbj_net = NetworkSubject(record_probe = my_probe, network_name = 'alexnet')
sbj_net._network.eval()
# print(sbj_net.layer_names) #print to see layer names for the probe


#define a scorer
n_dict = {l: range(10)} # list of neurons to be scored
scorer = MaxActivityScorer(trg_neurons = n_dict, aggregate=lambda x: np.stack(list(x.values())))

#initialize the optimizer
optim = GeneticOptimizer(
    states_shape=gen.input_dim,
    random_distr='normal',
    mutation_rate=0.3,
    mutation_size=0.3,
    population_size=population_size,
    temperature=1.0,
    num_parents=2,
)

# produce initial (stimuli, msg)
opt_state = optim.init()


#optimization loop
num_gens = 1000
rec_dict = defaultdict(list) #record dict will save layerwise as a list the activations of interest of the subject
best_nat =0; best_gen = 0 #initialize the best scores for nat and gen imgs

progress = trange(num_gens, desc='Generation 0 | best score: --- | avg score: --- ')
for g in progress:
    # Use current optimizer states to produce new images
    stimuli, msg = gen(opt_state, mask=mask)
    
    # Convert stimuli to subject state using the subject
    sub_state, msg =  sbj_net(data=(stimuli,msg))
    
    #update the keys of rec_dict
    for k, v in sub_state.items():
        rec_dict[k].append(v)

    # Use scorer to score the newly computed subject states
    stm_score, msg = scorer(
        data=(sub_state, msg)
    )
    
    #get for both nat and gen imgs maximal activation and argmax (the following lines can  be implemented more elegantly)
    max_gen, max_nat = logicwise_function(f = [np.amax, np.argmax], np_arr= stm_score[0], np_l= msg.mask)
    #update the best images
    if max_gen[0] > best_gen:
        best_gen = max_gen[0]
        best_gen_img = stimuli[msg.mask][max_gen[1]]
    if max_nat[0] > best_nat:
        best_nat = max_nat[0]
        best_nat_img = stimuli[~msg.mask][max_nat[1]]
        

    # Use the score to step the optimizer
    opt_state = optim.step(
        data=(stm_score[0], msg)
    )
    
    # Get the current best score to update the progress bar
    stat = optim.stats
    best = cast(NDArray, stat['best_score']).mean()
    curr = cast(NDArray, stat['curr_score']).mean()
    desc = f'Generation {g} | best score: {best:.1f} | avg score: {curr:.1f}'
    progress.set_description(desc)
    


#simple plotting of the output (Ponce 2019 style)
import matplotlib.pyplot as plt

fix, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(optim.stats['best_shist'], label='Synthetic')
ax[0].plot(optim.stats_nat['best_shist'], label='Natural')
ax[0].set_xlabel('Generation cycles')
ax[0].set_ylabel('Max Target Activations')
ax[0].set_title('Better than Ponce...')
ax[0].legend()

ax[1].plot(optim.stats['mean_shist'])
ax[1].plot(optim.stats_nat['mean_shist'])
ax[1].set_xlabel('Generation cycles')
ax[1].set_ylabel('Avg Target Activations')
ax[1].set_title('... and Kreimann')
ax[1].legend()

plt.show()

#see best nat and gen image
to_pil_image(best_gen_img)

to_pil_image(best_nat_img)

print(best_nat, best_gen)


