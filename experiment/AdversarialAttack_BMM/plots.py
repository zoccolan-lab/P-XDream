import os
import matplotlib.pyplot as plt
import seaborn as sns

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
