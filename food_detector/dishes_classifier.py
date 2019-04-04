import os
import numpy as np 
import cv2

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T 
import torch.nn.functional as F
from collections import defaultdict, Counter

feature=None

def get_infer_feature(module,inputs,ouput):
    global feature
    feature= F.normalize(inputs[0],p=2,dim=1)

class feature_extractor:
    def __init__(self,cfg):
        self.weight = cfg.EXTRACTOR.WEIGHT
        self.model = models.resnet50(pretrained=False)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts,136)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.load_weight()

        self.model.eval()
        self.model.module.fc.register_forward_hook(get_infer_feature)

        self.transforms = self.build_transform()




    def load_weight(self):
        checkpoint = torch.load(self.weight)
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def build_transform(self):
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ]
        )
        return transform
    
    def pre_process(self,imgs,boxes):
        imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
        batch_imgs=[]
        for box in boxes:
            box = box.to(torch.int64)
            top_left,bottom_right = box[:2].tolist(),box[2:].tolist()
            roi = imgs[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
            patch=self.transforms(roi)
            batch_imgs.append(patch)
        return torch.stack(batch_imgs,dim=0)
    
    def get_out_put(self,imgs,boxes):
        process_imgs = self.pre_process(imgs,boxes)
        with torch.no_grad():
            img_cuda = process_imgs.cuda(0,non_blocking=True)
            ouput = self.model(img_cuda)
        return feature



