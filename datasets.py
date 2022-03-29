#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: xiangli
"""
import numpy as np
import cv2
from torch.utils.data import Dataset

class Pair(Dataset):
    def __init__(self,seqs,transforms=None,pairs_per_seq=1):
        super(Pair,self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))
        self.return_mata = getattr(seqs,'return_meta',False)
    
    def  __get__item(self,index):
        index = self.indices[index%len(self.indices)]
        img_files, anno, meta = self.seqs[index]
        frame_indices = list(range(len(img_files))) #This is the indices of video frames for a specific sequence
        rand_z, rand_x = np.sort(np.random.choice(frame_indices,2,replace=False))
        while rand_x-rand_z > 100: 
            rand_z, rand_x = np.sort(np.random.choice(frame_indices,2,replace=False))# The two chosen frames should be at most T frames apart 
        z = cv2.imread(img_files[rand_z],cv2.IMREAD_COLOR) #May need to be converted to RGB color space
        x = cv2.imread(img_files[rand_x],cv2.IMREAD_COLOR) #May need to be converted to RGB color space
        box_z = anno[rand_z]
        box_x = anno[rand_x]
        item = (z,x,box_z,box_x)
        item = self.transforms(item)
        return item
    
    def __len__(self):
        return len(self.indices)*self.pairs_per_seq
    
        
        
        
    

