from pathlib import Path
import os
    
def get_net_paths(base_nets_dir='/content/drive/MyDrive/XDREAM'):
    """
    Retrieves the paths of the files of the weights of pytorch neural nets within a base directory and returns a dictionary
    where the keys are the file names and the values are the full paths to those files.

    Args:
        base_nets_dir (str): The path of the base directory (i.e. the dir that contains all nn files). Default is '/content/drive/MyDrive/XDREAM'.

    Returns:
        Dict[str, str]: A dictionary where the keys are the nn file names and the values are the full paths to those files.
    """
    nets_dict = {'base_nets_dir':Path(base_nets_dir)}
    for root, _, files in os.walk(base_nets_dir):
        for f in files:
          if f.lower().endswith(('.pt', '.pth')):
            file_path = os.path.join(root, f)
            nets_dict[f] = file_path
    return nets_dict



