# ConceptLens
This is the source code for paper "What’s Pulling the Strings? Evaluating Integrity and Attribution in AI Training and Inference through Concept Shift". The code is partly based on albef code https://github.com/salesforce/ALBEF.

Our algorithm propose a generic framework that leverages pre-trained multimodal models to identify the root causes of vulnerabilities in these three dimensions by analyzing conceptual shifts in probing samples. 

## Requirements
pytorch==1.8.0
transformers==4.8.1
timm==0.4.9
We also provide a requirements.txt file, run 
```
conda create --name <env> --file requirements.txt
```
to create your environment automatically.

## Dataset
### Data Integrity
#### Data poisoning and bias injection.
- Dog Samples in SBU Captions Dataset
  
#### Privacy exposure
- Mnist
- Cifar10

### Model Integrity
#### Adversarial perturbations
For Unimodel classfier probing samples, we use
- Mnist
- Cifar10
- CelebA

For Multimodel vision language pre-training model probing samples, we use
- Flickr
- MSCOCO
- RefCoco+
- SNLI-VE

#### Toxic and biased generation
- Generated samples form dreambooth

### Probing Sample 
Please put all samples needed to analyze in ```\probing_samples```, for uni-model task, the adv samples, org samples, adv labels, org samples, and org predict labels is needed. We release probing samples collected by us in https://drive.google.com/drive/folders/1esbdww3fvjgRZ3ttboymd668z1idR4Bg?usp=sharing, and provide the cifar sample with fgsm attack with this repo for demo.

We provide demo code for Uni-model extraction. 



## Pretrained Model 
Please download the 14M pre-trained albef model from: https://github.com/salesforce/ALBEF?tab=readme-ov-file

## Usage
### Feature extraction
To extract features for probing samples above 3 tasks, run: 
```
mkdir "results/[NAME_PROBING_SAMPLE]"

python feature_extract.py --gpu [GPU-NUMBER] --cls --config [PATH_TO_CONFIG_FILE]  --checkpoint [PATH_TO_ALBEF_WEIGHT] --dataset_name [DATASET] --adv_name [NAME_PROBING_SAMPLE] --type_data ['multi' or 'uni']
```

Here is an example to use the probing sample we provide in  ```\probing_samples``` with 5 files:  cifar_org.npy (original samples), cifar_labels_org.npy (original labels for those samples),cifar_labels_pre.npy (model prediction for original samples),  cifar_fgsm_0.0625.npy (attack_samples), cifar_fgsm_0.0625_labels.npy (model prediction for attack samples). To extract features to analyze fgsm attack on cifar dataset, run:
```
mkdir "results/cifar_fgsm_0.0625"

python feature_extract.py --gpu 0 --cls --config configs/concept_prism.yaml  --checkpoint ALBEF.pth --dataset_name cifar --adv_name cifar_fgsm_0.0625 --type_data uni
```

After running these commands, the extract features metrics will be generate under ```/results/[NAME_PROBING_SAMPLE]```

### Analysis and Visualization
Here we provide several tools to analyze the extract features with visualization
#### Linear abstract feature similarity
To generate the diagram visualizing the semantic shift of probing sample, run
```
python visualize_feature_similarity.py --adv_name [NAME_PROBING_SAMPLE] 

### Visualize cifar_fgsm probing samples
python visualize_feature_similarity.py --adv_name cifar_fgsm_0.0625 
```
The diagram will be generate: ```/visual/[NAME_PROBING_SAMPLE]_feature_sim.pdf```
#### Attention difference
To generate the heatmap of the attention map and the cross-attention map, run
```
python visualize_attention_difference.py --adv_name [NAME_PROBING_SAMPLE] --type_data ['multi' or 'uni']

### Visualize cifar_fgsm probing samples
python visualize_attention_difference.py --adv_name cifar_fgsm_0.0625 --type_data uni
```
It will generate 6 attention map under  ```/visual"```
#### Detection
To get the results of detection for unimodal attacks, run
```
python single_modality_detect.py --adv_name [NAME_PROBING_SAMPLE] 

### Detecting cifar_fgsm probing samples
python single_modality_detect.py --adv_name cifar_fgsm_0.0625
```
The detection results will be printed out.
We will release the detection tool for other tasks soon.

