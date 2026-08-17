"""Step 1 data preparation: convert UEyes into the folder layout train.py expects
(data_ueyes/{images,maps,fixation_maps}/{train,val}/), so SaliconDataset can load
it exactly like it loads SALICON.

- images/: copied as-is, original extension preserved (UEyes mixes jpg/png/jpeg).
- maps/: the final 0-7s aggregate heatmap, re-saved as .png from UEyes'
  saliency_maps/heatmaps_7s (a smooth/continuous map, so the lossy jpg source for
  ~35% of images is not a meaningful quality concern here).
- fixation_maps/: NOT copied from UEyes' fixmaps_7s. For images whose source was
  .jpg, that file is itself a lossy JPEG re-encoding of a sparse binary dot
  pattern, which introduces compression ringing around each fixation point and
  corrupts the strict 0/255 structure the loss functions (NSS in particular)
  expect. Instead this script rebuilds a clean binary fixation map directly
  from the raw per-fixation coordinates in eyetracker_logs/, via ueyes_utils.
  This is the same log-parsing building block Step 2 (temporal volumes) will
  reuse with a binned time window instead of the full 0-7s window used here.

  Rasterized directly at TARGET_SIZE (matching SaliconDataset's fixed 256x256
  working resolution), not at each image's native resolution. SaliconDataset
  resizes gt to 256x256 (cv2.resize) but never resizes fixations, relying on
  every fixation map already sharing one common size -- true for SALICON
  (always 640x480) but not for UEyes, whose source screenshots have all sorts
  of native resolutions/aspect ratios. Rasterizing straight onto the 256x256
  grid also avoids losing isolated fixation points to a later downsampling
  pass over an already-sparse map.
"""
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ueyes_utils import load_all_fixations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UEYES_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'UEyes_dataset')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_ueyes')
TARGET_SIZE = 256


def main():
    image_types = pd.read_csv(os.path.join(UEYES_DIR, 'image_types.csv'), sep=';')
    image_types['ImageId'] = image_types['Image Name'].apply(lambda n: os.path.splitext(n)[0])
    raw_split_of = dict(zip(image_types['ImageId'], image_types['Train/Test']))

    images_dir = os.path.join(UEYES_DIR, 'images')
    heatmaps_dir = os.path.join(UEYES_DIR, 'saliency_maps', 'heatmaps_7s')
    logs_dir = os.path.join(UEYES_DIR, 'eyetracker_logs')

    image_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(images_dir)
        if not f.startswith('.')
    }
    print(f'Found {len(image_files)} images in {images_dir}')

    missing_split = [i for i in image_files if i not in raw_split_of]
    if missing_split:
        print(f'WARNING: {len(missing_split)} images had no Train/Test entry in image_types.csv, defaulting to train')
    split_of = {
        image_id: ('val' if raw_split_of.get(image_id) == 'Test' else 'train')
        for image_id in image_files
    }

    for split in ('train', 'val'):
        for sub in ('images', 'maps', 'fixation_maps'):
            os.makedirs(os.path.join(OUT_DIR, sub, split), exist_ok=True)

    # --- images/ + maps/ ---
    for image_id, fname in tqdm(image_files.items(), desc='Copying images + maps'):
        split = split_of[image_id]

        src_img = os.path.join(images_dir, fname)
        dst_img = os.path.join(OUT_DIR, 'images', split, fname)
        shutil.copy(src_img, dst_img)

        src_heatmap = os.path.join(heatmaps_dir, fname)
        dst_heatmap = os.path.join(OUT_DIR, 'maps', split, image_id + '.png')
        Image.open(src_heatmap).convert('L').save(dst_heatmap)

    # --- fixation_maps/, rebuilt from raw eyetracker logs ---
    fixations = load_all_fixations(logs_dir)
    grouped = fixations.groupby('MEDIA_NAME')
    print(f'{len(fixations)} valid fixations loaded, covering {fixations["MEDIA_NAME"].nunique()} unique images')

    n_no_fixations = 0
    for image_id, fname in tqdm(image_files.items(), desc='Building fixation maps'):
        split = split_of[image_id]

        fixmap = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        if fname in grouped.groups:
            img_fix = grouped.get_group(fname)
            xs = (img_fix['FPOGX'].to_numpy() * TARGET_SIZE).astype(int)
            ys = (img_fix['FPOGY'].to_numpy() * TARGET_SIZE).astype(int)
            valid = (xs >= 0) & (xs < TARGET_SIZE) & (ys >= 0) & (ys < TARGET_SIZE)
            fixmap[ys[valid], xs[valid]] = 255
        else:
            n_no_fixations += 1

        dst_fix = os.path.join(OUT_DIR, 'fixation_maps', split, image_id + '.png')
        Image.fromarray(fixmap, mode='L').save(dst_fix)

    if n_no_fixations:
        print(f'WARNING: {n_no_fixations} images had zero matching fixations in eyetracker_logs')

    print('Done. Output written to', OUT_DIR)


if __name__ == '__main__':
    main()
