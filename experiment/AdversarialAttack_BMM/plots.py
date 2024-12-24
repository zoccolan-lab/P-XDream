import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import UnivariateSpline

from pxdream.scorer import ParetoReferencePairDistanceScorer


plt.rcParams.update({
    'axes.labelsize': 30,      # Etichette degli assi
    'xtick.labelsize': 25,     # Ticks dell'asse X
    'ytick.labelsize': 25,     # Ticks dell'asse Y
    'legend.fontsize': 20,     # Testo della legenda
    'legend.title_fontsize': 25,  # Titolo della legenda
    'lines.markersize': 15
})

def BMM_scatter_plot(df, net_name = 'resnet50_r', savepath = '', include_density = False):

    # Crea una colormap
    cmap = plt.get_cmap('jet')

    # Mappa le categorie a colori
    categories = df['high_target'].unique()
    category_colors = {category: cmap(i / len(categories)) for i, category in enumerate(categories)}

    # Crea lo scatter plot
    plt.figure(figsize=(15, 8))

    scatter_plots = []
    for task, marker in [('adversarial', 'o'), ('invariance', '*')]:
        # Aggiungi il density plot se richiesto
        task_data = df[df['task'] == task]
        x = task_data['dist_low'].abs()
        y = (task_data['dist_up'].abs() / task_data['ref_activ']) * 100
        scat_col = task_data['high_target'].map(category_colors)
        scat_alpha = 1
        if include_density:
            scat_col = 'k'
            plt.rcParams.update({'lines.markersize': 3})
            sns.kdeplot(x=x,y=y,
                levels=5,
                cmap='viridis',
                shade=True,
                alpha=1
            )
        scatter = plt.scatter(x=x,y=y,
            c=scat_col,
            label=task,
            marker=marker,
            alpha=0.1
        )
        scatter_plots.append(scatter)
        

    # Aggiungi etichette e titolo
    plt.xlabel('Pixel distance')
    plt.ylabel('Activation distance \n(% of reference)')
    robust = '' if '_r' in net_name else 'not '
    net_only = net_name.split("_")[0].capitalize()
    plt.title(f'{net_only} pretrained - {robust}robust', fontsize=35)
    plt.legend(title='Task')
    plt.xlim([0,300])
    plt.ylim([0,120])

    # Mostra il plot
    if savepath:
        plt.savefig(os.path.join(savepath,f'scatterBMM_{net_name}.svg'), format='svg')


def avg_spline(x, y, nbins=20, n_points_spline=200, savepath: str|None=None):
    bins = np.linspace(x.min(), x.max(), nbins+1)
    digitized = np.digitize(x, bins)
    # Calculate means and standard deviations
    bin_means_x = []
    bin_means_y = []
    bin_stds_y = []
    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:  # if bin not empty
            bin_means_x.append(x[mask].mean())
            bin_means_y.append(y[mask].mean())
            bin_stds_y.append(y[mask].std())

    bin_means_x = np.array(bin_means_x)
    bin_means_y = np.array(bin_means_y)
    bin_stds_y = np.array(bin_stds_y)      
    
    # Fit splines
    spl = UnivariateSpline(bin_means_x, bin_means_y, k=3, s=0.1)
    spl_upper = UnivariateSpline(bin_means_x, bin_means_y + bin_stds_y, k=3, s=0.1)
    spl_lower = UnivariateSpline(bin_means_x, bin_means_y - bin_stds_y, k=3, s=0.1)

    # Create smooth points for plotting
    x_smooth = np.linspace(x.min(), x.max(), n_points_spline)
    y_smooth = spl(x_smooth)
    y_upper = spl_upper(x_smooth)
    y_lower = spl_lower(x_smooth)

    # Plot
    if savepath:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.2, label='Original data')
        plt.scatter(bin_means_x, bin_means_y, color='red', s=50, label='Bin means')
        plt.plot(x_smooth, y_smooth, 'r-')
        plt.fill_between(x_smooth, y_lower, y_upper, alpha=0.2, color='red', label='±1 std')
        plt.legend()
        plt.xlabel('\u0394 Pixel')
        plt.ylabel('\u0394 Neuron activation (%)')
        plt.savefig(savepath, format='png', dpi=450, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    return {'spline': spl,'spl_upper': spl_upper, 'spl_lower': spl_lower, 'xbounds': (x.min(), x.max())}
    
def plot_metaexp_p1(p1_all, savepath = None):
    lys = list(p1_all.keys())
    x = abs(p1_all[lys[0]])
    y = abs(p1_all[lys[1]])
    spline = avg_spline(x, y, savepath=savepath)
    return spline

def pf1_fromPKL(data, exp_id: int, plot_data: bool = False):
    ref_val = data['reference_activ'][exp_id]
    p1 = data['p1_front'][exp_id]
    p1_it = p1[:,0]
    p1_el = p1[:,1]
    ly_scores = data['layer_scores'][exp_id]
    #check that points are all in pf1
    p1_pts = {k:v[p1_it, p1_el] for k,v in ly_scores.items()}
    pf_vec , coordinates = ParetoReferencePairDistanceScorer.pareto_front(p1_pts, 
                                        weights = [-1,1], first_front_only=True)
    if np.unique(pf_vec).size != 1 or pf_vec[0] != 0:
        print('Not all points are in the first pareto front')
        p1_it = p1_it[coordinates].astype(np.int32)
        p1_el = p1_el[coordinates].astype(np.int32)
        
    lys = list(ly_scores.keys())
    ly_down = ly_scores[lys[0]]
    ly_up = ly_scores[lys[1]] #156_linear_73
    p1_down = ly_down[p1_it, p1_el]
    p1_up = (ly_up[p1_it, p1_el]/ref_val)*100

    ly_down = ly_down.flatten()
    ly_up = (ly_up.flatten()/ref_val)*100


    # Plot the scatter with gradually decreasing alpha
    if plot_data:
        plt.scatter(ly_down, ly_up, alpha=np.linspace(0.1, 1, len(ly_down)))
        plt.scatter(p1_down, p1_up, color='red', alpha=np.linspace(0.1, 1, len(p1_down)))
        plt.xlabel('\u0394 Pixel')
        plt.ylabel('\u0394 Neuron activation (%)')
    return {lys[0]: p1_down, lys[1]: p1_up}