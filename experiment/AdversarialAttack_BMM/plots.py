import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.labelsize': 30,      # Etichette degli assi
    'xtick.labelsize': 25,     # Ticks dell'asse X
    'ytick.labelsize': 25,     # Ticks dell'asse Y
    'legend.fontsize': 20,     # Testo della legenda
    'legend.title_fontsize': 25,  # Titolo della legenda
    'lines.markersize': 15
})

def BMM_scatter_plot(df, net_name = 'resnet50_r', savepath = ''):
  # Carica il DataFrame

  # Crea una colormap
  cmap = plt.get_cmap('jet')

  # Mappa le categorie a colori
  categories = df['high_target'].unique()
  category_colors = {category: cmap(i / len(categories)) for i, category in enumerate(categories)}

  # Crea lo scatter plot
  plt.figure(figsize=(15, 8))

  scatter_plots = []
  for task, marker in [('adversarial', 'o'), ('invariance', '*')]:
      task_data = df[df['task'] == task]
      scatter = plt.scatter(
          task_data['dist_low'].abs(),
          (task_data['dist_up'].abs()/task_data['ref_activ'])*100,
          c=task_data['high_target'].map(category_colors),
          label=task,
          marker=marker
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

  # Mostra il plot
  if savepath:
    plt.savefig(os.path.join(savepath,f'scatterBMM_{net_name}.svg'), format='svg')
