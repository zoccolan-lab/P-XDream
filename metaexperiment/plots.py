
# --- PLOTTING ---
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import re

import numpy as np
import pandas as pd


# Step 1: Define the groups to plot
groups_to_plot = [
    {'task': 'invariance', 'train_type': 'vanilla', 'cmap':     plt.cm.get_cmap('Blues')},
    {'task': 'invariance', 'train_type': 'robust_l2', 'cmap':   plt.cm.get_cmap('Purples')},
    {'task': 'invariance', 'train_type': 'robust_linf', 'cmap': plt.cm.get_cmap('Greens')}
]

def natural_sort_key(s):
    """Extract numeric part from string for sorting"""
    if s == 'NaN':
        return float('inf')
    numbers = re.findall(r'\d+', str(s))
    return float(numbers[0]) if numbers else float('inf')


# Step 2: Plot each group
def metaplot_lines(grouped_stats_df: pd.DataFrame, groups_to_plot: dict[dict]):
    constraints = grouped_stats_df['constraint'].unique()
    desired_order = constraints[np.argsort([natural_sort_key(x) for x in constraints])]

    for group in groups_to_plot:
        query = ' & '.join([f"{k} == '{v}'" for k,v in group.items() if k in grouped_stats_df.columns])
        #query = f"task == '{group['task']}' & train_type == '{group['train_type']}'"
        subset_df = grouped_stats_df.query(query)
        # Convert 'constraint' to a categorical type with the specified order
        subset_df['constraint'] = pd.Categorical(subset_df['constraint'], categories=desired_order, ordered=True)
        # Sort the DataFrame
        subset_df = subset_df.sort_values('constraint').reset_index(drop=True)
                
        x = subset_df['dist_low'].apply(lambda x:x['mean'])
        y = subset_df['dist_up_perc'].apply(lambda x:x['mean'])
        
        alphas = np.linspace(0.25, 1, len(y))
        colors = [group['cmap'](t) for t in alphas] 
        
        for _x, _y, c in zip(subset_df['dist_low'], subset_df['dist_up_perc'], colors):
            print(_x)
            plt.errorbar(
                x=abs(_x['mean']),
                y=abs(_y['mean']),
                xerr=_x['std'],
                yerr=_y['std'],
                marker='none',
                color=c,
                linestyle='None',
                zorder=10,
            )
        
        plt.scatter(
            x = x,
            y = y,
            label='_nolegend_',
            marker='o',
            linestyle='None',
            fc=colors,
            ec='black',
            s=50,
            zorder=10,
        )
        # Create a proxy artist with the final color for the legend
        plt.scatter([], [], label=f"{group['train_type']}", marker='o',
                    fc=colors[-1], ec='black', s=50)
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if len(segments) > 0:
            # Subdivide each segment into 100 steps
            subdivided_segments = []
            for seg in segments:
                x_vals = np.linspace(seg[0][0], seg[1][0], 101)
                y_vals = np.linspace(seg[0][1], seg[1][1], 101)
                subdivided = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
                subdivided_segments.append(np.concatenate([subdivided[:-1], subdivided[1:]], axis=1))
            subdivided_segments = np.concatenate(subdivided_segments, axis=0)

            # Create a LineCollection with progressive alpha
            total_segments = len(subdivided_segments)
            alphas = np.linspace(0.25, 1, total_segments)
            colors = [group['cmap'](t) for t in alphas]  # Blue color with varying alpha
            lc = LineCollection(subdivided_segments, colors=colors, linewidth=2)

            # Add the LineCollection to the plot
            plt.gca().add_collection(lc)

    plt.xlim(0, 388)
    plt.ylim(0, 100)
    # Step 3: Customize the plot
    plt.xlabel('\u0394 pixel')
    plt.ylabel('\u0394 neuron activation (%)')
    plt.title('Varying constraints to invariance experiments')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.grid(False)

    plt.show()

#usage: metaplot_lines(grouped_stats_df, groups_to_plot)

#TO BE IMPROVED:

def plot_grouped_splines(grouped_splines: dict, comparison_key: str = 'adversarial', 
                        fig_size: tuple = (12, 6)) -> None:
    """Plot splines for different model groups with consistent styling."""
    plt.figure(figsize=fig_size)

    # Create color maps
    reds = plt.cm.Reds(np.linspace(0.2, 1, 10))
    blues = plt.cm.Blues(np.linspace(0.2, 1, 10))
    color_idx = 0

    for group_name, group_data in grouped_splines.items():
        nn_type, tt, it, c = group_name.split('#')
        color_map = plt.cm.get_cmap('tab10')
        linestyle_map = ['-', '--', '-.', ':']
        color_intensity = np.linspace(0.2, 1, 10)

        # Assign color based on nn_type
        color = color_map(hash(nn_type) % 10)

        # Assign linestyle based on tt
        linestyle = linestyle_map[hash(tt) % len(linestyle_map)]

        # Adjust color intensity based on c
        color = (color[0], color[1], color[2], color_intensity[hash(c) % len(color_intensity)])
        if comparison_key in group_data:
            # Style based on robustness
            # if 'robust' in group_name.lower():
            #     color = reds[color_idx % len(reds)]
            #     linestyle = '--'
            # else:
            #     color = blues[color_idx % len(blues)]
            #     linestyle = '-'

            # Extract and clean model name
            start = group_name.find("net")
            clean_name = group_name[start:] if start != -1 else group_name

            # Get spline data
            bounds = group_data[comparison_key]['xbounds']
            spline = group_data[comparison_key]['spline']
            
            # Evaluate spline
            x_eval = np.linspace(bounds[0], bounds[1], 1000)
            y_eval = spline(x_eval)

            plt.plot(x_eval, y_eval, label=clean_name, 
                    color=color, linestyle=linestyle)
            color_idx += 1

    # Format plot
    plt.xlabel('\u0394 Pixel')
    plt.ylabel('\u0394 Neuron activation (%)')
    plt.ylim(0, 120)
    plt.legend(bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', 
              ncol=3)