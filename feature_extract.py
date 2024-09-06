import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.model_pretrain import ALBEF as ALBEF_mlm
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

from transformers import BertForMaskedLM
import cv2
import utils

from torchvision import transforms

from dataset import pair_dataset, pair_dataset_classifer,pair_dataset_vlp
from PIL import Image
from torchvision import transforms

print('load package')

def feature_extract(model, ref_model, data_loader, tokenizer, device, config,model_mlm,args):
    # test
    model.eval()
    ref_model.eval()
    start_time = time.time()
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset)
    image_feats = torch.zeros(num_image, config['embed_dim'])
    image_embeds = torch.zeros(num_image, 257, 768)
    text_feats = torch.zeros(num_text, config['embed_dim'])
    text_embeds = torch.zeros(num_text, 30, 768)
    text_atts = torch.zeros(num_text, 30).long()
    gradcams = torch.zeros(num_text, 30, 16, 16)
    att_cams = torch.zeros(num_text, 30, 16, 16)
    text_inputs = torch.zeros(num_text, 30).long()
    success_images = torch.zeros(num_image, 3, 256, 256)
    if args.type_data == 'multi' or args.type_data == 'a2b' or args.type_data == 'mi':
        mlms = torch.zeros(num_text, 30, 30522)
    else:
        mlms = torch.zeros(num_text, 30522)
    # cross_atts = torch.zeros(len(data_loader.dataset.ann), 30, 768)
    for inputimages, adv_images, texts,adv_text, texts_ids in data_loader:
        images = inputimages.to(device)
        texts_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(device)
        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        images = images_normalize(images)
        output = model.inference(images, texts_input, use_embeds=False)
        model.zero_grad()
        output['loss'].backward()  
        mlm = model_mlm.mlm(images, texts_input, args.type_data)
        with torch.no_grad():
            mask = texts_input.attention_mask.view(texts_input.attention_mask.size(0),1,-1,1,1)
            grads=model.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attn_gradients()
            cams=model.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attention_map()
            
            cams = cams[:, :, :, 1:].reshape(images.size(0), 12, -1, 16, 16) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(images.size(0), 12, -1, 16, 16) * mask
            gradcam = cams * grads
            
            image_feats[texts_ids] = output['image_feat'].cpu().detach()
            image_embeds[texts_ids] = output['image_embed'].cpu().detach()
            success_images[texts_ids] = inputimages
            print(texts_ids)
            text_feats[texts_ids] = output['text_feat'].cpu().detach()
            text_embeds[texts_ids] = output['text_embed'].cpu().detach()
            
            
            text_atts[texts_ids] = texts_input.attention_mask.cpu().detach()
            gradcams[texts_ids] =  gradcam.mean(1).cpu().detach()
            att_cams[texts_ids] = cams.mean(1).cpu().detach()
            text_inputs[texts_ids] = texts_input.input_ids.cpu().detach()
            mlms[texts_ids] = mlm.cpu().detach()
            
        
        
    torch.save(image_feats,f'results/{args.adv_name}/org_image_feats.pth')
    torch.save(image_embeds,f'results/{args.adv_name}/org_image_embeds.pth')
    torch.save(text_feats,f'results/{args.adv_name}/org_text_feats.pth')
    torch.save(text_embeds,f'results/{args.adv_name}/org_text_embeds.pth')
    torch.save(text_atts,f'results/{args.adv_name}/org_text_atts.pth')
    torch.save(gradcams,f'results/{args.adv_name}/org_gradcams.pth')
    torch.save(att_cams,f'results/{args.adv_name}/org_att_cams.pth')
    torch.save(text_inputs,f'results/{args.adv_name}/org_texts_inputs.pth')
    torch.save(success_images,f'results/{args.adv_name}/success_images.pth')
    torch.save(mlms,f'results/{args.adv_name}/org_mlms.pth')
    
    print('Forward')
    
    image_feats = torch.zeros(num_image, config['embed_dim'])
    image_embeds = torch.zeros(num_image, 257, 768)

    text_feats = torch.zeros(num_text, config['embed_dim'])
    text_embeds = torch.zeros(num_text, 30, 768)
    text_atts = torch.zeros(num_text, 30).long()
    text_inputs = torch.zeros(num_text, 30).long()
    att_cams = torch.zeros(num_text, 30, 16, 16)
    gradcams = torch.zeros(num_text, 30, 16, 16)
    success_images = torch.zeros(num_image, 3, 256, 256)
    
    if args.type_data == 'multi' or args.type_data == 'a2b'  or args.type_data == 'mi':
        mlms = torch.zeros(num_text, 30, 30522)
    else:
        mlms = torch.zeros(num_text, 30522)
    
    for images, adv_images,texts, adv_text,texts_ids in data_loader:
        images = adv_images.to(device)
        texts_input = tokenizer(adv_text, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(device)
        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        images = images_normalize(images)
        output = model.inference(images, texts_input, use_embeds=False)
        model.zero_grad()
        output['loss'].backward()  
        mlm = model_mlm.mlm(images, texts_input, args.type_data)
        with torch.no_grad():
            mask = texts_input.attention_mask.view(texts_input.attention_mask.size(0),1,-1,1,1)
            grads=model.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attn_gradients()
            cams=model.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attention_map()
            cams = cams[:, :, :, 1:].reshape(images.size(0), 12, -1, 16, 16) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(images.size(0), 12, -1, 16, 16) * mask
            print(mask)
            gradcam = cams * grads
            image_feats[images_ids] = output['image_feat'].cpu().detach()
            image_embeds[images_ids] = output['image_embed'].cpu().detach()
            text_feats[texts_ids] = output['text_feat'].cpu().detach()
            text_embeds[texts_ids] = output['text_embed'].cpu().detach()
            text_atts[texts_ids] = texts_input.attention_mask.cpu().detach()
            gradcams[texts_ids] =  gradcam.mean(1).cpu().detach()
            att_cams[texts_ids] =  cams.mean(1).cpu().detach()
            text_inputs[texts_ids] = texts_input.input_ids.cpu().detach()
            success_images[texts_ids] = adv_images
            mlms[texts_ids] = mlm.cpu().detach()
            

    torch.save(image_feats,f'results/{args.adv_name}/adv_image_feats.pth')
    torch.save(image_embeds,f'results/{args.adv_name}/adv_image_embeds.pth')
    torch.save(text_feats,f'results/{args.adv_name}/adv_text_feats.pth')
    torch.save(text_embeds,f'results/{args.adv_name}/adv_text_embeds.pth')
    torch.save(text_atts,f'results/{args.adv_name}/adv_text_atts.pth')
    torch.save(gradcams,f'results/{args.adv_name}/adv_gradcams.pth')
    torch.save(att_cams,f'results/{args.adv_name}/adv_att_cams.pth')
    torch.save(text_inputs,f'results/{args.adv_name}/adv_texts_inputs.pth')
    torch.save(success_images,f'results/{args.adv_name}/adv_success_images.pth')
    torch.save(mlms,f'results/{args.adv_name}/adv_mlms.pth')
    
    
    
    return



def main(args, config):
    device = args.gpu[0]

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    if args.type_data == 'multi':
        test_dataset = pair_dataset_vlp(test_transform, args.adv_name, args.dataset_name)
    elif args.type_data == 'uni' or args.type_data == 'a2b' or args.type_data == 'mi':
        test_dataset = pair_dataset_classifer(test_transform, args.adv_name, args.dataset_name)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=4)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model_mlm = ALBEF_mlm(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    ### load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint
    msg2 = model_mlm.load_state_dict(state_dict, strict=False)
    for key in list(state_dict.keys()):
        print(key)
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    
    
    print('msg', msg)
    print('msg2', msg2)

    print('load checkpoint from %s' % args.checkpoint)
    # print(msg)

    model = model.to(device)
    model_mlm = model_mlm.to(device)
    ref_model = ref_model.to(device)

    print("Start eval")
    start_time = time.time()


    feature_extract(model, ref_model, test_loader, tokenizer, device, config, model_mlm,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/concept_prism.yaml')
    parser.add_argument('--checkpoint', default='ALBEF.pth')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+', default=0)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--adv_name',  default='./')
    parser.add_argument('--dataset_name',  default='./')
    parser.add_argument('--type_data',  default='uni')
    

    args = parser.parse_args()
 
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)



