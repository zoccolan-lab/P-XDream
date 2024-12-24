import os
import re
import lpips
import torch
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


#Quick LPIPS start coming from the LPIPS github page https://github.com/richzhang/PerceptualSimilarity
#for additional details see original LPIPS paper https://arxiv.org/abs/1801.03924
#import lpips
#loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
#loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
#
#import torch
#img0 = torch.ones(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
#img1 = torch.ones(1,3,64,64)
#d = loss_fn_alex(img0, img1)


FOLDER_PATH = '/home/ltausani/Desktop/Datafolders/Image_selections/Clustering/InetAll_top200_avg_subtract_BelowNyquist_sf002to02_rep3/Im_selection'
LPIPS_NET = 'vgg'
INET_COLOR = True

# Funzione per caricare un'immagine, convertirla in RGB e normalizzarla nell'intervallo [-1, 1]
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0  # Normalizza a [0, 1]
    img = (img * 2) - 1  # Normalizza a [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def calculate_lpips(lpips_dist, img1_path, img2_path):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    dist = lpips_dist(img1, img2)
    return dist.item()

def nat_imgs_lpips(folder_path, lpips_net='alex', inet_color = False):
    # Inizializza la funzione di perdita LPIPS con il modello AlexNet
    lpips_dist = lpips.LPIPS(net=lpips_net)
    
    # Ottieni la lista dei sottocartelle
    subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Ottieni la lista di tutti i file .png presenti in tutti i sottocartelle
    image_files = [os.path.join(subfolder, f) for subfolder in subfolders for f in os.listdir(subfolder) if f.endswith('.png')]
    
    
    #questo if è temporaneo
    if inet_color and all([os.path.basename(ifile)[0]=='n' for ifile in image_files]):
        inet_idxs = [int(re.search(r'inet(\d+)\.png$', p).group(1)) for p in image_files]
        metrics = pd.read_csv(os.path.join(os.path.dirname(folder_path), 'metrics.csv'))
        image_files = metrics[metrics['idxs'].isin(inet_idxs)]['path2orig'].values
        col_str = 'RGB'
    else:
        col_str = 'Gray'
    # Crea una lista per memorizzare le distanze
    distances_list = []

    # Calcola LPIPS tra ogni coppia di immagini diverse e memorizza nella lista
    for i in tqdm(range(len(image_files)), desc="Processing images"):
        for j in range(i + 1, len(image_files)):
            img1_path = image_files[i]
            img2_path = image_files[j]

            # Aggiungi la distanza alla lista
            distances_list.append({
                'Image1': img1_path,
                'Image2': img2_path,
                'LPIPS': calculate_lpips(lpips_dist, img1_path, img2_path)
            })
            
    # Crea un DataFrame dalle distanze
    distances_df = pd.DataFrame(distances_list)

    # Salva il DataFrame come file CSV
    csv_path = os.path.join(folder_path, f'lpips_{lpips_net}_{col_str}.csv')
    distances_df.to_csv(csv_path, index=False)

def SnS_lpips(path2sns_exp:str, lpips_net:str='alex'):
    lpips_dist = lpips.LPIPS(net=lpips_net)
    
    impath = os.path.join(path2sns_exp, 'images')
    
    data_summary = pd.read_csv(os.path.join(path2sns_exp, 'data_summary.csv'))
    lpips_d = {'LPIPS': []}
    for i in data_summary.index.values:
        ref = os.path.join(impath, f'ref_{i}.png')
        sns_gen = os.path.join(impath, f"{data_summary['task'][i]}_{i}.png")
        lpips_d['LPIPS'].append(calculate_lpips(lpips_dist, ref, sns_gen))
    data_summary['LPIPS'] = np.array(lpips_d['LPIPS'])
    return data_summary

if __name__ == "__main__":
    nat_imgs_lpips(folder_path = FOLDER_PATH, lpips_net = LPIPS_NET, inet_color = INET_COLOR)