import json
import os
import random

from torch.utils.data import Dataset
import numpy as np
import torch

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index


class pair_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            for j, caption in enumerate(ann['caption']):
                self.image.append(ann['image'])
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text = self.text[index]

        return image, text, index


class pair_dataset_vlp(Dataset):
    def __init__(self, transform, adv_name, dataset_name, max_words=30):
        # We will release this part of code soon
        pass
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        curimage = (self.image[index] * 255).astype(np.uint8)
        image = Image.fromarray(curimage)
        image = self.transform(image)
        
        cur_attack_image = (self.attack_image[index] * 255).astype(np.uint8)
        
        attack_image = Image.fromarray(cur_attack_image)
        attack_image = self.transform(attack_image)
        
        text = self.text[index]
        attack_text = self.attack_text[index]
        return image, attack_image, text, attack_text, index

class pair_dataset_classifer(Dataset):
    def __init__(self, transform, adv_name, dataset_name, max_words=30):
        self.transform = transform
        self.max_words = max_words

        self.text = []
        self.image = []
        self.adv_image = []
        self.adv_text = []

        self.txt2img = {}
        self.img2txt = {}

        
        txt_id = 0
        
        images = np.load(f'probing_samples/{dataset_name}_org.npy')
        attack_images = np.load(f'probing_samples/{adv_name}.npy')
        labels = np.load(f'probing_samples/{dataset_name}_labels_org.npy')
        attack_labels = np.load(f'probing_samples/{adv_name}_labels.npy')
        pre_labels = np.load(f'probing_samples/{dataset_name}_labels_pre.npy')
        if dataset_name == 'cifar':
            text_labels = ['an airplane with the body, wings around the sky', 'an automobile', 'a bird', 'a cat', 'a deer', 'a dog', 'a frog', 'a horse', 'a ship', 'a truck']
        if dataset_name == 'mnist':
            text_labels = ['letter zero','letter one', 'letter two', 'letter three', 'letter four', 'letter five', 'letter six', 'letter seven', 'letter eight', 'letter nine']
            images = images.astype('float')
            images = np.repeat(images, 3, axis=-1)
            attack_images = attack_images.astype('float')
            attack_images = np.repeat(attack_images, 3, axis=-1)
        if dataset_name == 'celeba':
            text_labels = ['a man', 'a woman']
            
        image_num = 0
        for i in range(10000):
            if (labels[i] == pre_labels[i] and labels[i] != attack_labels[i]):
                self.img2txt[image_num] = []

                self.image.append(images[i])
                self.adv_image.append(attack_images[i])
                
                caption = f'This is an image of {text_labels[labels[i]]}'
                adv_caption = f'This is an image of {text_labels[labels[i]]}'
                self.text.append(pre_caption(caption, self.max_words))
                self.adv_text.append(pre_caption(adv_caption, self.max_words))
                self.txt2img[txt_id] = image_num
                self.img2txt[image_num].append(txt_id)
                txt_id += 1
                image_num += 1
            if image_num == 500:
                break


    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        curimage = (self.image[index] * 255).astype(np.uint8)
        
        image = Image.fromarray(curimage)
        image = self.transform(image)
        
        cur_attack_image = (self.adv_image[index] * 255).astype(np.uint8)
        adv = curimage - cur_attack_image
        attack_image = Image.fromarray(cur_attack_image)
        attack_image = self.transform(attack_image)
        
        text = self.text[index]
        adv_text = self.adv_text[index]

        return image, attack_image, text, adv_text, index


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            
    @property
    def text(self):
        t = []
        for ann in self.ann:
            t += ann['caption']
        return t
