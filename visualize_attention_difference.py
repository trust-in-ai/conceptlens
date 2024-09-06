import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.stats import skew, kurtosis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.covariance import EllipticEnvelope
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse


def filter_non_black_maps(attention_maps):
    non_black_indices = (attention_maps.view(attention_maps.size(0), attention_maps.size(1), -1).sum(dim=2) != 0)
    filtered_maps = attention_maps[non_black_indices].view(-1, 16, 16)
    return filtered_maps

def filter_different_words_attention_maps(org_texts, adv_texts, org_map, adv_map):
    different_org_map = []
    different_adv_map = []
    for i in range(org_texts.shape[0]):  
        for j in range(org_texts.shape[1]): 
            if org_texts[i, j] != adv_texts[i, j]: 
                different_org_map.append(org_map[i, j])
                different_adv_map.append(adv_map[i, j])
    return torch.mean(torch.stack(different_org_map),dim=0), torch.mean(torch.stack(different_adv_map),dim=0)

def draw(adv_name, type_data):
    sns.set()
    if type_data == 'uni':
        adv_grad_cams = torch.mean(filter_non_black_maps(torch.load(f'results/{adv_name}/adv_gradcams.pth')[:,7]),dim=0)
        org_grad_cams = torch.mean(filter_non_black_maps(torch.load(f'results/{adv_name}/org_gradcams.pth')[:,7]),dim=0)
        adv_att_cams = torch.mean(filter_non_black_maps(torch.load(f'results/{adv_name}/adv_att_cams.pth')[:,7]),dim=0)
        org_att_cams = torch.mean(filter_non_black_maps(torch.load(f'results/{adv_name}/org_att_cams.pth')[:,7]),dim=0)
    elif type_data == 'multi':
        adv_grad_cams = torch.load(f'results/{adv_name}/adv_gradcams.pth')
        org_grad_cams = torch.load(f'results/{adv_name}/org_gradcams.pth')
        adv_att_cams = torch.load(f'results/{adv_name}/adv_att_cams.pth')
        org_att_cams = torch.load(f'results/{adv_name}/org_att_cams.pth')
        adv_text = torch.load(f'results/{adv_name}/adv_texts_inputs.pth')
        org_text = torch.load(f'results/{adv_name}/org_texts_inputs.pth')
        org_att_cams, adv_att_cams = filter_different_words_attention_maps(
        org_text, adv_text, org_att_cams, adv_att_cams)
        org_grad_cams, adv_grad_cams = filter_different_words_attention_maps(
        org_text, adv_text, org_grad_cams, adv_grad_cams)
    difference_grad = org_grad_cams - adv_grad_cams
    difference_att = org_att_cams - adv_att_cams

    
    vmin_grad = min(torch.min(adv_grad_cams).item(), torch.min(org_grad_cams).item())
    vmax_grad = max(torch.max(adv_grad_cams).item(), torch.max(org_grad_cams).item())
    
    vmin_att = min(torch.min(adv_att_cams).item(), torch.min(org_att_cams).item())
    vmax_att = max(torch.max(adv_att_cams).item(), torch.max(org_att_cams).item())

    heatmaps_data = [
    (org_att_cams, f'{adv_name}_org_att', vmin_att, vmax_att),
    (adv_att_cams, f'{adv_name}_adv_att', vmin_att, vmax_att),
    (difference_att, f'{adv_name}_diff_att', None, None),
    (org_grad_cams, f'{adv_name}_org_grad', vmin_grad, vmax_grad),
    (adv_grad_cams, f'{adv_name}_adv_grad', vmin_grad, vmax_grad),
    (difference_grad, f'{adv_name}_diff_grad', None, None)
    ]

    # Loop through the data and generate heatmaps
    for data, filename, vmin, vmax in heatmaps_data:
        plt.figure(figsize=(6, 5))
        if vmin is None or vmax is None:
            vmin = data.numpy().min()
            vmax = data.numpy().max()
    
        ax = sns.heatmap(data.numpy(), cmap='viridis', xticklabels=False, yticklabels=False, cbar=True,cbar_kws={"shrink": 0.6, "aspect": 10,"pad": 0.02}, vmin=vmin, vmax=vmax)
        
        # Customize the colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, vmax])  # Only show top and bottom values
        cbar.ax.yaxis.set_tick_params(labelsize=24)  # Adjust padding to reduce white space
        cbar.ax.yaxis.get_offset_text().set_size(14)  # Make the offset text (if any) larger
        cbar.ax.yaxis.get_offset_text().set_weight('bold')  # Make the offset text bold
        cbar.set_ticklabels([f'{vmin:.4f}', f'{vmax:.4f}'])  # Adjust tick labels precision
        
        # Remove any extra space between the heatmap and the colorbar
        cbar.ax.locator_params(nbins=2)  # Make sure only two ticks are shown, minimizing gaps
        
        # Rotate the colorbar tick labels vertically
        cbar.ax.yaxis.set_tick_params(rotation=270)

        # Use tight layout and save with no white border
        plt.axis('off')  # Turn off axis if needed
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove space around the heatmap
        plt.savefig(f'visual/{filename}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)  # Save without white borders
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--adv_name',  default='./')
    parser.add_argument('--type_data',  default='uni')
    

    args = parser.parse_args()

    draw(args.adv_name, args.type_data)

