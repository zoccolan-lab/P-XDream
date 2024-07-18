
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

out_dir = os.path.join(OUT_DIR, "neuron_scaling_fit_function")

FIT_FROM   = 1
FIG_SUB    = ( 2,  2)
FIG_SIZE   = (20, 10)
POINT_SIZE = 50
LINE_WIDTH = 2
PALETTE    = 'husl'

def fit_curve(
    points    : List[Tuple[float, float]],
    function  : Callable[[float, ...], float],  # type: ignore
    init_coef : List[float]         | None = None,
    domain    : Tuple[float, float] | None = None,
    fit_from  : int                        = 0,
    logger    : Logger | SilentLogger      = SilentLogger()
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
):
    FIG_X, FIG_Y = FIG_SUB
    
    custom_palette = sns.color_palette(PALETTE, len(functions))
    
    for single_plot, log_scale in itertools.product([False, True], [False, True]):
        
        fig, ax = plt.subplots(figsize=FIG_SIZE) if single_plot else plt.subplots(FIG_X, FIG_Y, figsize=FIG_SIZE)

        for i, (layer, fun) in enumerate(functions.items()):
                        
            x = [x_ for x_,  _ in points[layer] ]
            y = [y_ for  _, y_ in points[layer] ]
            
            col = custom_palette[i]
            ax_ = ax if single_plot else ax[i // 2][i % 2] 
            
            ax_.scatter(x, y, color=col, marker='.', s=POINT_SIZE)
            if not single_plot: ax_.set_title(layer)
            
            start, end = 1, LAYER_SETTINGS[layer]['neurons']
            
            x_pred = range(start, end+1)
            y_pred = [fun(x_) for x_ in x_pred]
            
            ax_.plot(x_pred, y_pred, color=col, linestyle='--', label=layer, linewidth=LINE_WIDTH)
            
            # Plot with original x scale
            if log_scale: ax_.set_xscale('log')
                
        if single_plot: ax_.legend()
        
        scale_str  = 'log'    if log_scale   else 'orig'
        single_str = 'single' if single_plot else 'multi'        
        out_fp = os.path.join(out_dir, f'fit_{single_str}_{scale_str}scale.svg')
        
        logger.info(f'Saving plot to {out_fp}')
        fig.savefig(out_fp)
        

def main():
    
    # Logger and directories
    logger = LoguruLogger(on_file=False)
    
    fit_dir = make_dir(
        path=os.path.join(out_dir, os.path.basename(NEURON_SCALING_FILE).split('.')[0]),
        logger=logger
    )

    # Load data
    neuron_scaling = load_pickle(NEURON_SCALING_FILE)
    
    # Plot optimization curves
    plot_optimizing_units(
        data=neuron_scaling['score'],
        out_dir=fit_dir,
        logger=logger
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
    plot_fit(points=points, functions=out_functions, logger=logger, out_dir=fit_dir)
    
    # Save functions
    out_fp = os.path.join(fit_dir, 'functions.pkl')
    logger.info(f'Saving functions to {out_fp}')
    
    store_pickle(out_functions, out_fp)

if __name__ == '__main__': main()
