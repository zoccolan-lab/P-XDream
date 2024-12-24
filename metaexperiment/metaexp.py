from dataclasses import dataclass
import os
import pickle
from typing import Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pprint
import argparse

from experiment.AdversarialAttack_BMM.plots import pf1_fromPKL, plot_metaexp_p1
from pxdream.utils.io_ import load_pickle
from tqdm import tqdm

def agg_stats(x: pd.Series) -> dict:
    """
    Calculates the mean and standard error of the mean (SEM) for a series of data.
    :param x: The data series for which to calculate the statistics.
    :type x: pandas.Series
    :return: A dictionary containing the mean and SEM of the data.
    :rtype: dict
    """
    if x.dtype in ['float64', 'int64']:
        return {
            'mean': abs(x.mean()),
            'std': x.std(),
            'sem': x.std()/np.sqrt(len(x))
        }
    return {
        'mean': x.iloc[0],
    }


def merge_data_pkl(dict_list: list[dict]) -> dict:
    """Merge a list of dictionaries preserving original data types for each key."""
    if not dict_list:
        return {}
    if len(dict_list) == 1:
        return dict_list[0]
        
    result = dict_list[0]
    for next_dict in dict_list[1:]:
        merged = {}
        common_keys = set(result.keys()) & set(next_dict.keys())
        
        for key in common_keys:
            val1, val2 = result[key], next_dict[key]
            
            if isinstance(val1, list):
                merged[key] = val1 + val2
            elif isinstance(val1, str):
                merged[key] = val1 + "\n" + val2
            else:
                raise TypeError(f"Unsupported type {type(val1)} for key {key}")
                
        result = merged
        
    return result

@dataclass
class SnS_metadata:
    """Hierarchical structure for neural network experiments data"""
    data: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]

    @staticmethod
    def _find_file(dir_path: Union[str, Path], file_type = "*.csv") -> Path:
        """Find the first CSV file in a directory"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{dir_path} is not a directory")
            
        csv_files = list(dir_path.glob(file_type))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {dir_path}")
            
        return csv_files
    
    @classmethod
    def from_json(cls, json_path: str) -> 'SnS_metadata':
        
        with open(json_path, 'r') as f:
            paths_tree = json.load(f)
        
        # Initialize nested dictionary
        data = {}
        
        # First level: Neural Network type
        for nn_type, nn_data in tqdm(paths_tree.items(), desc="Neural Network Types"):
            data[nn_type] = {}
            
            # Second level: Model type
            for model_type, model_data in tqdm(nn_data.items(), desc="Model Types", leave=False):
                data[nn_type][model_type] = {}
                
                # Third level: Iterations
                for iterations, iter_data in tqdm(model_data.items(), desc="Iterations", leave=False):
                    data[nn_type][model_type][iterations] = {}
                    
                    # Fourth level: Constraints
                    for constraint, dir_paths in tqdm(iter_data.items(), desc="Constraints", leave=False):
                        if isinstance(dir_paths, str):
                            dir_paths = [dir_paths]
                        data[nn_type][model_type][iterations][constraint] = {}
                        data[nn_type][model_type][iterations][constraint]['path'] = dir_paths
                        # Load and concatenate CSVs
                        dfs = []
                        data_pkl = []
                        for dir_path in tqdm(dir_paths, desc="Directory Paths", leave=False):
                            try:
                                csv_path = cls._find_file(dir_path, file_type="*.csv")[0]
                                df = pd.read_csv(csv_path)
                                dfs.append(df)
                                data_pkl.append(load_pickle(os.path.join(dir_path, 'data.pkl')))
                                
                            except (NotADirectoryError, FileNotFoundError) as e:
                                print(f"Warning: {e}")
                                continue
                        
                        if dfs:
                            data[nn_type][model_type][iterations][constraint]['df'] = pd.concat(dfs, ignore_index=True)
                            metadata_sess = data[nn_type][model_type][iterations][constraint]['df']
                            mexp_data = merge_data_pkl(data_pkl)
                            data[nn_type][model_type][iterations][constraint]['splines'] = {}
                            for t in tqdm(metadata_sess['task'].unique(), desc="Tasks", leave=False):
                                p1 = []
                                task_ds = metadata_sess[metadata_sess['task'] == t]
                                for i in task_ds.index:
                                    p1.append(pf1_fromPKL(mexp_data, i, plot_data=False))
                                p1_all = {
                                    key: np.hstack([d[key] for d in p1]).flatten()
                                    for key in p1[0].keys()
                                }
                                spl = plot_metaexp_p1(p1_all, savepath=os.path.join(dir_path, f'{t}_avg_p1.png'))
                                data[nn_type][model_type][iterations][constraint]['splines'][t] = spl
                                
                        else:
                            print(f"Warning: No valid data found for {nn_type}/{model_type}/{iterations}/{constraint}")
        
        return cls(data)
        
    def update_from_json(self, json_path: str) -> None:
        """
        Update existing metadata with new data from json file, loading only missing entries
        """
        # Load new paths tree from JSON
        with open(json_path, 'r') as f:
            paths_tree = json.load(f)
        
        # First level: Neural Network type
        for nn_type, nn_data in tqdm(paths_tree.items(), desc="Neural Network Types"):
            if nn_type not in self.data:
                self.data[nn_type] = {}
                
            # Second level: Model type
            for model_type, model_data in tqdm(nn_data.items(), desc="Model Types", leave=False):
                if model_type not in self.data[nn_type]:
                    self.data[nn_type][model_type] = {}
                    
                # Third level: Iterations
                for iterations, iter_data in tqdm(model_data.items(), desc="Iterations", leave=False):
                    if iterations not in self.data[nn_type][model_type]:
                        self.data[nn_type][model_type][iterations] = {}
                        
                    # Fourth level: Constraints
                    for constraint, dir_paths in tqdm(iter_data.items(), desc="Constraints", leave=False):
                        # Skip if constraint already exists
                        if constraint in self.data[nn_type][model_type][iterations]:
                            continue
                            
                        # Handle single path or list of paths
                        if isinstance(dir_paths, str):
                            dir_paths = [dir_paths]
                        self.data[nn_type][model_type][iterations][constraint] = {}
                        self.data[nn_type][model_type][iterations][constraint]['path'] = dir_paths
                        
                        # Load and concatenate CSVs
                        dfs = []
                        data_pkl = []
                        for dir_path in tqdm(dir_paths, desc="Directory Paths", leave=False):
                            try:
                                csv_path = self._find_file(dir_path, file_type="*.csv")[0]
                                df = pd.read_csv(csv_path)
                                dfs.append(df)
                                data_pkl.append(load_pickle(os.path.join(dir_path, 'data.pkl')))
                                
                            except (NotADirectoryError, FileNotFoundError) as e:
                                print(f"Warning: {e}")
                                continue
                        
                        if dfs:
                            self.data[nn_type][model_type][iterations][constraint]['df'] = pd.concat(dfs, ignore_index=True)
                            metadata_sess = self.data[nn_type][model_type][iterations][constraint]['df']
                            mexp_data = merge_data_pkl(data_pkl)
                            self.data[nn_type][model_type][iterations][constraint]['splines'] = {}
                            for t in tqdm(metadata_sess['task'].unique(), desc="Tasks", leave=False):
                                p1 = []
                                task_ds = metadata_sess[metadata_sess['task'] == t]
                                for i in task_ds.index:
                                    p1.append(pf1_fromPKL(mexp_data, i, plot_data=False))
                                p1_all = {
                                    key: np.hstack([d[key] for d in p1]).flatten()
                                    for key in p1[0].keys()
                                }
                                spl = plot_metaexp_p1(p1_all, savepath=os.path.join(dir_path, f'{t}_avg_p1.png'))
                                self.data[nn_type][model_type][iterations][constraint]['splines'][t] = spl
                                
                        else:
                            print(f"Warning: No valid data found for {nn_type}/{model_type}/{iterations}/{constraint}")
                    
    def get_experiment(self, nn_type: str, model_type: str, iterations: str, constraint: str) -> pd.DataFrame:
        """Retrieve specific experiment data"""
        return self.data[nn_type][model_type][iterations][constraint]
    
    @property
    def tree_structure(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Hierarchical structure of the experiments data"""
        tree = {}
        for nn_type, nn_data in self.data.items():
            tree[nn_type] = {}
            for model_type, model_data in nn_data.items():
                tree[nn_type][model_type] = {}
                for iterations, iter_data in model_data.items():
                    tree[nn_type][model_type][iterations] = list(iter_data.keys())
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(tree)
        return tree

    def save_pkl(self, output_path: Union[str, Path], protocol: Optional[int] = None) -> None:
        """Save metadata to a pickle file"""
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.data, f, protocol=protocol)
        except Exception as e:
            raise IOError(f"Error saving metadata to {output_path}: {str(e)}")

    @classmethod
    def load_pkl(cls, input_path: Union[str, Path]) -> 'SnS_metadata':
        """Load metadata from a pickle file"""
        input_path = Path(input_path)
        
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            return cls(data=data)
        except Exception as e:
            raise IOError(f"Error loading metadata from {input_path}: {str(e)}")
        


    def aggregate(self, nn=None, train_types=None, n_it=None):
        """Aggregate experiment statistics with flexible parameter selection.
        
        Args:
            nn: Neural network type (str or None)
            train_types: List of training types (list or None) 
            n_it: Number of iterations (str or None)
        
        Returns:
            pd.DataFrame: Aggregated statistics
        """
        # Get all possible values if parameter is None
        nn_types = [nn] if nn else list(self.tree_structure.keys())
        grouped_stats_df = []
        grouped_splines = {}
        for nn_type in nn_types:
            train_type_list = [train_types] if train_types else list(self.tree_structure[nn_type].keys())
            
            for tt in train_type_list:
                n_it_list = [n_it] if n_it else list(self.tree_structure[nn_type][tt].keys())
                
                for it in n_it_list:
                    for c in self.tree_structure[nn_type][tt][it]:
                        data = self.get_experiment(
                            nn_type=nn_type,
                            model_type=tt, 
                            iterations=it,
                            constraint=c
                        )
                        df = data['df']
                        spline_key = '#'.join([nn_type, tt, it, c])
                        grouped_splines[spline_key] = data['splines']
                        
                        if 'dist_up_perc' not in df.columns:
                            df['dist_up_perc'] = abs(df['dist_up']/df['ref_activ'])*100
                            
                        grouped_stats = df.groupby(['task']).agg(agg_stats)[['dist_low', 'dist_up_perc']]
                        grouped_stats['net'] = nn_type
                        grouped_stats['train_type'] = tt
                        grouped_stats['n_it'] = it
                        grouped_stats['constraint'] = c
                        grouped_stats = grouped_stats.reset_index()
                        grouped_stats_df.append(grouped_stats)
        
        return pd.concat(grouped_stats_df, ignore_index=True), grouped_splines
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and save SnS metadata.")
    parser.add_argument("--json_path", type=str, default = "SnS_multiexp_dirs.json", help="Path to the JSON file containing the metadata paths.")
    parser.add_argument("--output_path", type=str, default = "metaexp.pkl", help="Path to save the output pickle file.")
    
    args = parser.parse_args()
    
    # Load metadata from JSON
    metadata = SnS_metadata.from_json(args.json_path)
    
    # Save metadata to pickle file
    metadata.save_pkl(args.output_path)