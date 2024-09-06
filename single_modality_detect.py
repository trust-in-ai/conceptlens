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
import argparse


def detect(adv_name):

    adv_att_cams = torch.load(f'results/{adv_name}/adv_att_cams.pth').numpy()[:,7,:,:]
    adv_att_cams = adv_att_cams.reshape(len(adv_att_cams),-1)

    org_att_cams = torch.load(f'results/{adv_name}/org_att_cams.pth').numpy()[:,7,:,:]
    org_att_cams = org_att_cams.reshape(len(org_att_cams),-1)

    adv_grad_cams = torch.load(f'results/{adv_name}/adv_gradcams.pth').numpy()[:,7,:,:]
    adv_grad_cams = adv_grad_cams.reshape(len(adv_grad_cams),-1)

    org_grad_cams = torch.load(f'results/{adv_name}/org_gradcams.pth').numpy()[:,7,:,:]
    org_grad_cams = org_grad_cams.reshape(len(org_grad_cams),-1)


    org_image_feature = torch.load(f'results/{adv_name}/org_image_feats.pth')
    org_text_feature = torch.load(f'results/{adv_name}/org_text_feats.pth')
    adv_image_feature = torch.load(f'results/{adv_name}/adv_image_feats.pth')
    adv_text_feature = torch.load(f'results/{adv_name}/adv_text_feats.pth')

    adv_sims_matrix = adv_image_feature @ adv_text_feature.t()
    adv_sims_matrix = torch.diag(adv_sims_matrix).numpy()
    org_sims_matrix = org_image_feature @ org_text_feature.t()
    org_sims_matrix = torch.diag(org_sims_matrix).numpy()
    org_image_feature = org_image_feature.numpy()
    adv_image_feature = adv_image_feature.numpy()


    adv_combined_vector = np.hstack((adv_sims_matrix[:, np.newaxis], adv_image_feature, adv_att_cams, adv_grad_cams))
    org_combined_vector = np.hstack((org_sims_matrix[:, np.newaxis], org_image_feature, org_att_cams, org_grad_cams))

    
    
    X_orignal = org_combined_vector  
    X_attack =  adv_combined_vector
    # values = ''

    envelope = EllipticEnvelope(contamination=0.01)
    envelope.fit(X_orignal)
    e_preds = envelope.predict(X_attack)
    accuracy_e = (e_preds == -1).mean()
    fpr_e= (envelope.predict(X_orignal) == -1).mean()
    # values += f'{accuracy_e},{fpr_e},'
    print('dr: ', accuracy_e, ' fpr: ',fpr_e)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--adv_name',  default='./')
    

    args = parser.parse_args()

    detect(args.adv_name)