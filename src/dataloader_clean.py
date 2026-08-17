from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, cv2
from utils import *
import json
import random
from pycocotools.coco import COCO

   
class SaliconDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png', vol_dir=None, time_slices=5):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.exten = exten
        # vol_dir is optional: when set, __getitem__ also loads the per-slice
        # temporal saliency volume (Step 2's saliency_volumes_5) and returns a
        # 4-tuple instead of 3. Needed for pnas_boosted_multi, irrelevant for
        # the plain pnas model, which has no temporal branch to supervise.
        self.vol_dir = vol_dir
        self.time_slices = time_slices
        # UEyes mixes .jpg/.png/.jpeg source images (SALICON is always .jpg),
        # so look up each image's real extension instead of assuming .jpg.
        self.img_filenames = {
            os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
        }
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, self.img_filenames[img_id])
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256,256))
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        fixations = np.array(Image.open(fix_path).convert('L'))
        fixations = fixations.astype('float')
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0

        if self.vol_dir is None:
            return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

        # Volumes are saved at generation-time resolution (640x480, see
        # generate_volumes_ueyes.py), not pre-resized to 256x256 like
        # fixation_maps -- resize explicitly here, same as gt above, instead
        # of repeating the Step 1 bug where fixations were never resized.
        vol = np.zeros((self.time_slices, 256, 256), dtype='float32')
        for t in range(self.time_slices):
            slice_path = os.path.join(self.vol_dir, img_id + '_' + str(t) + self.exten)
            s = np.array(Image.open(slice_path).convert('L')).astype('float32')
            s = cv2.resize(s, (256, 256))
            if s.max() > 1.0:
                s = s / 255.0
            vol[t] = s

        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations), torch.FloatTensor(vol)

    def __len__(self):		
         return len(self.img_ids)
