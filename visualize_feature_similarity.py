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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def draw(adv_name):
    sns.set()
    adv_image_feature = torch.load(f'results/{adv_name}/adv_image_feats.pth')
    adv_text_feature = torch.load(f'results/{adv_name}/adv_text_feats.pth')

    org_image_feature = torch.load(f'results/{adv_name}/org_image_feats.pth')
    org_text_feature = torch.load(f'results/{adv_name}/org_text_feats.pth')
    
    org_image_feature = org_image_feature.reshape(len(org_image_feature),-1)
    adv_image_feature = adv_image_feature.reshape(len(adv_image_feature),-1)

    adv_sims_matrix = adv_image_feature @ adv_text_feature.t()
    adv_sims_matrix = torch.diag(adv_sims_matrix).numpy()
    org_sims_matrix = org_image_feature @ org_text_feature.t()
    org_sims_matrix = torch.diag(org_sims_matrix).numpy()

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(6, 5))
    plt.hist(org_sims_matrix, bins=50, alpha=0.5, label='Original', color='b', density=True)
    plt.hist(adv_sims_matrix, bins=50, alpha=0.5, label='Perturbed', color='r',  density=True)
    sns.kdeplot(org_sims_matrix, color='b', linewidth=2)  # Blue for Original
    sns.kdeplot(adv_sims_matrix, color='r', linewidth=2) 
    plt.legend(fontsize=20, loc='upper right', ncol=1)
    plt.xticks(fontsize=20, rotation = -25)
    plt.yticks(fontsize=20)
    # plt.ylim(0,50)
    plt.ylabel('Frequency',fontsize=20)
    plt.xlabel('Similarity',fontsize=20)
    plt.savefig(f'visual/{adv_name}_feature_sim.pdf', dpi=300, bbox_inches='tight')

    plt.show()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--adv_name',  default='./')
    

    args = parser.parse_args()

    draw(args.adv_name)

