
import itertools
import os
import inspect
import numpy as np
from typing     import Callable, Dict, List, Tuple
from statistics import mean

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

from analysis.utils.misc import CurveFitter
from analysis.utils.settings import OUT_DIR, NEURON_SCALING_FILE, LAYER_SETTINGS
from experiment.MaximizeActivity.plot import plot_optimizing_units
from experiment.utils.misc import make_dir
from zdream.utils.io_        import load_pickle, store_pickle
from zdream.utils.logger     import Logger, LoguruLogger, SilentLogger
from zdream.utils.misc       import default

# --- SETTINGS ---

out_dir = os.path.join(OUT_DIR, "neuron_scaling")

FIT_FROM = 1


def fit_curve(
    points    : List[Tuple[float, float]],
    function  : Callable[[float, ...], float],  # type: ignore
    init_coef : List[float]         | None = None,
    domain    : Tuple[float, float] | None = None,
    fit_from  : int                        = 0,
    logger    : Logger | SilentLogger      = SilentLogger(),
) -> CurveFitter:
    '''
    Fit a curve to the given data points using the given function.

    :param points: Points to fit the curve.
    :type points: List[Tuple[float, float]]
    :param function: Function to fit the curve, the first parameter is the x value of points
        while the rest are the parameters to fit.
    :type function: Callable[[Tuple[float, ...]], float]
    :param init_coef: Initial coefficient as a start to the fitting process, defaults to None
    :type init_coef: List[float] | None, optional
    :param domain: Domain under which the function is fitted, defaults to None
    :type domain: Tuple[float, float] | None, optional
    :param fit_from: Point to start fitting to, defaults to 0
    :type fit_from: int, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    '''
    
    # Parameters
    parameters = list(inspect.signature(function).parameters.keys())[1:]
    default(init_coef, [1] * len(parameters))
    
    # Extract data points
    x_data     = [x for x, _ in points]
    y_data     = [y for _, y in points]
    x_data_fit = x_data[fit_from:]
    y_data_fit = y_data[fit_from:]
    
    # Fit the curve
    popt, pcov = curve_fit(function, x_data_fit, y_data_fit)
    logger.info(f"Function fitted to {len(points) - fit_from} points with parameters:")
    for p, o in zip(parameters, popt):
        logger.info(f"{p} = {round(o, 5)}")
    logger.info(f"")
    
    return CurveFitter(function, popt, domain)

def plot_fit(
    points: Dict[str, List[Tuple[float, float]]],
    functions: Dict[str, Callable[[float], float]],
    logger: Logger | SilentLogger = SilentLogger(),
    out_dir: str = '.',
    **kwargs
):
    # --- MACROS ---

    FIG_SUB    = kwargs.get('FIG_SUB', (2, 2))  # Grid layout (rows, columns)
    FIG_SIZE   = kwargs.get('FIG_SIZE', (20, 9))
    TITLE      = kwargs.get('TITLE', 'Fit Plot')
    X_LABEL    = kwargs.get('X_LABEL', 'Random optimized units')
    Y_LABEL    = kwargs.get('Y_LABEL', 'Fitness')
    SAVE_FORMATS = kwargs.get('SAVE_FORMATS', ['svg'])

    PALETTE    = kwargs.get('PALETTE', 'husl')
    PLOT_ARGS  = kwargs.get('PLOT_ARGS', {'linestyle': '--', 'linewidth': 3, 'alpha': 0.6})
    GRID_ARGS  = kwargs.get('GRID_ARGS', {'linestyle': '--', 'alpha': 0.7})

    # Font customization
    FONT               = kwargs.get('FONT', 'serif')
    TICK_ARGS          = kwargs.get('TICK_ARGS', {'labelsize': 14, 'direction': 'out', 'length': 6, 'width': 2, 'labelfontfamily': FONT})
    LABEL_ARGS         = kwargs.get('LABEL_ARGS', {'fontsize': 16, 'labelpad': 10, 'fontfamily': FONT})
    TITLE_ARGS         = kwargs.get('TITLE_ARGS', {'fontsize': 20, 'fontweight': 'bold', 'fontfamily': FONT})
    SUBPLOT_TITLE_ARGS = kwargs.get('SUBPLOT_TITLE_ARGS', {'fontsize': 16, 'fontfamily': FONT})  # New arguments for subplot titles
    SCATTER_ARGS       = kwargs.get('SCATTER_ARGS', {'marker': '.', 's': 90})
    LEGEND_ARGS        = kwargs.get('LEGEND_ARGS', {
        'frameon': True, 'fancybox': True, 
        'framealpha': 0.7, 'loc': 'best', 'prop': {'family': FONT, 'size': 14}
    })

    FIG_X, FIG_Y = FIG_SUB

    FIG_X, FIG_Y = FIG_SUB

    custom_palette = sns.color_palette(PALETTE, len(functions))
    
    for single_plot, log_scale in itertools.product([False, True], [False, True]):
        
        fig, ax = plt.subplots(figsize=FIG_SIZE) if single_plot else plt.subplots(FIG_X, FIG_Y, figsize=FIG_SIZE)

        for i, (layer, fun) in enumerate(functions.items()):

            if layer not in points:
                logger.warn(f"Layer '{layer}' is missing in points. Skipping.")
                continue
                        
            x = [x_ for x_,  _ in points[layer]]
            y = [y_ for  _, y_ in points[layer]]
            
            col = custom_palette[i]
            ax_ = ax if single_plot else ax[i // FIG_Y][i % FIG_Y]
            
            ax_.scatter(x, y, color=col, **SCATTER_ARGS)
            if not single_plot: 
                ax_.set_title(LAYER_SETTINGS[layer]['title'], **SUBPLOT_TITLE_ARGS)  # Use subplot-specific title args
            
            start, end = 1, LAYER_SETTINGS[layer]['neurons']
            
            x_pred = range(start, end + 1)
            y_pred = [fun(x_) for x_ in x_pred]
            
            ax_.plot(x_pred, y_pred, color=col, label=LAYER_SETTINGS[layer]['title'], **PLOT_ARGS)
            
            # Plot with original x scale
            if log_scale: 
                ax_.set_xscale('log')
                
            # Set x-labels only on the last row
            if i // FIG_Y == FIG_X - 1:
                ax_.set_xlabel(X_LABEL, **LABEL_ARGS)

            # Set y-labels only on the first column
            if i % FIG_Y == 0:
                ax_.set_ylabel(Y_LABEL, **LABEL_ARGS)
                
            # Add grid
            ax_.grid(True, **GRID_ARGS)
            # Customize ticks
            ax_.tick_params(**TICK_ARGS)

        if single_plot:
            ax_.legend(**LEGEND_ARGS)

        fig.suptitle(f"{TITLE} - {'Log Scale' if log_scale else 'Original Scale'}", **TITLE_ARGS)
        
        # Save the plots in multiple formats
        scale_str  = 'log'    if log_scale   else 'orig'
        single_str = 'single' if single_plot else 'multi'        

        for fmt in SAVE_FORMATS:
            out_fp = os.path.join(out_dir, f'fit_{single_str}_{scale_str}scale.{fmt}')
            logger.info(f'Saving plot to {out_fp}')
            fig.savefig(out_fp, bbox_inches='tight')

def main():
    
    # Logger and directories
    logger = LoguruLogger(on_file=False)

    neuron_scaling_dir = os.path.dirname(NEURON_SCALING_FILE)
    neuron_scaling_file, ext = os.path.basename(NEURON_SCALING_FILE).split('.')
    variant_name = [part for part in neuron_scaling_file.split('_') if 'variant' in part][0].strip('variant')
    
    fit_dir = make_dir(
        path=os.path.join(out_dir, variant_name),
        logger=logger
    )

    # Load data
    neuron_scaling = load_pickle(NEURON_SCALING_FILE)
    
    # Plot optimization curves
    plot_optimizing_units(
        data=neuron_scaling['score'],
        out_dir=fit_dir,
        logger=logger,
        TITLE=F'{variant_name} Generator Variant - Neuron Optimization Scaling'
    )

    # Average samples
    points = {
        layer: [(neuron, mean([v[0] for v in vs])) for neuron, vs in scores.items()]
        for layer, scores in neuron_scaling['score'].items()
    }
    
    # Fit functions for each layer
    out_functions = {}

    for layer, point in points.items():
        
        logger.info(f'Fitting function for layer {layer}')
        
        domain = (1, LAYER_SETTINGS[layer]['neurons'])
        
        out_functions[layer] = fit_curve(
            points   = point,
            function = CurveFitter.hyperbolic_quadratic,  # type: ignore
            domain   = domain,
            fit_from = FIT_FROM,
            logger   = logger
        )
    
    # Plot the fit
    plot_fit(points=points, functions=out_functions, logger=logger, out_dir=fit_dir, TITLE=f"{variant_name} Generator Variant - Neuron Optimization Scaling Fit")
    
    # Save functions

    out_fp = os.path.join(neuron_scaling_dir, 'neuron_scaling_functions.pkl')

    if os.path.exists(out_fp):
        logger.info(f'Previous functions were computed. Adding just computed to existing file.')
        data = load_pickle(out_fp)
        data[variant_name] = out_functions
    else:
        data = {variant_name: out_functions}
    
    logger.info(f'Saving functions to {out_fp}')
    
    store_pickle(out_functions, out_fp)

if __name__ == '__main__': main()
