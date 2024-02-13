from pathlib import Path
import os
    
def get_net_paths(base_nets_dir='/content/drive/MyDrive/XDREAM'):
  nets_dict = {'base_nets_dir':Path(base_nets_dir)}
  for root, _, files in os.walk(base_nets_dir):
      for f in files:
          file_path = os.path.join(root, f)
          nets_dict[f] = file_path
  return nets_dict



