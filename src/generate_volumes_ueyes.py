"""Step 2: generate the 5-slice (0-1s, 1-2s, ..., 4-5s) temporal saliency ground
truth for UEyes, the equivalent of TempSAL's own generate_volumes.py + the
parse_fixations/get_saliency_volume/GaussianBlur2D helpers in utils.py -- but
reimplemented rather than reused, for two reasons:

- eyetracker_logs already has real per-fixation timestamps (FPOGS/FPOGD), so
  none of parse_fixations' heuristic timestamp-estimation (matching fixations
  to the nearest mouse-tracking sample, needed because SALICON's raw .mat
  files have no per-fixation timestamps) is needed here.
- utils.GaussianBlur2D is broken on any modern PyTorch: it calls F.conv1d on a
  5D tensor, and conv1d has only ever supported 2D/3D input -- verified
  empirically, it raises immediately regardless of CPU/GPU. Replaced with
  cv2.GaussianBlur (already used elsewhere in this codebase), same sigma=25 /
  kernel=201 constants as the original, applied per slice.

Binning rule: a fixation is assigned to bin floor(FPOGS), using only its start
time (never its duration) -- exactly generate_volumes.py's own rule. Anything
at or past TIME_SLICES seconds is clipped into the last bin rather than
dropped, again matching the original's min(..., time_slices-1); this is what
folds UEyes' longer ~7s trials into TempSAL's 5s/5-slice scheme without
discarding data.

Output canvas is a fixed 640x480 (utils.py's W/H) -- the same intermediate
resolution TempSAL itself works in for SALICON before the training-time
resize to 256x256, not each screenshot's native resolution. Resizing down to
256x256 is left to the Step 3 data loader, same as data_ueyes/maps already
works (see the note in TODO.md about the fixation_maps resize bug found in
Step 1 -- the Step 3 loader must not repeat it for these volumes).
"""
import os

import cv2
import numpy as np
from tqdm import tqdm

from ueyes_utils import load_all_fixations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UEYES_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'UEyes_dataset')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_ueyes')

W, H = 640, 480
TIME_SLICES = 5
SIGMA = 25
KSIZE = 201


def main():
    logs_dir = os.path.join(UEYES_DIR, 'eyetracker_logs')
    fixations = load_all_fixations(logs_dir)
    fixations['bin'] = np.minimum(fixations['FPOGS'].astype(int), TIME_SLICES - 1)
    grouped = fixations.groupby('MEDIA_NAME')

    for split in ('train', 'val'):
        for sub in ('fixation_volumes_5', 'saliency_volumes_5'):
            os.makedirs(os.path.join(OUT_DIR, sub, split), exist_ok=True)

    n_no_fixations = 0
    for split in ('train', 'val'):
        image_dir = os.path.join(OUT_DIR, 'images', split)
        image_files = sorted(os.listdir(image_dir))

        for fname in tqdm(image_files, desc=f'Generating volumes ({split})'):
            image_id = os.path.splitext(fname)[0]

            fix_vol = np.zeros((TIME_SLICES, H, W), dtype=np.float32)
            if fname in grouped.groups:
                img_fix = grouped.get_group(fname)
                xs = (img_fix['FPOGX'].to_numpy() * W).astype(int)
                ys = (img_fix['FPOGY'].to_numpy() * H).astype(int)
                bins = img_fix['bin'].to_numpy()
                valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                fix_vol[bins[valid], ys[valid], xs[valid]] = 1.0
            else:
                n_no_fixations += 1

            sal_vol = np.stack([
                cv2.GaussianBlur(fix_vol[t], (KSIZE, KSIZE), SIGMA)
                for t in range(TIME_SLICES)
            ])
            vmax = sal_vol.max()
            if vmax > 0:
                sal_vol = sal_vol / vmax

            for t in range(TIME_SLICES):
                fix_path = os.path.join(OUT_DIR, 'fixation_volumes_5', split, f'{image_id}_{t}.png')
                sal_path = os.path.join(OUT_DIR, 'saliency_volumes_5', split, f'{image_id}_{t}.png')
                cv2.imwrite(fix_path, (fix_vol[t] * 255).astype(np.uint8))
                cv2.imwrite(sal_path, (sal_vol[t] * 255).astype(np.uint8))

    if n_no_fixations:
        print(f'WARNING: {n_no_fixations} images had zero matching fixations in eyetracker_logs')

    print('Done. Output written to', OUT_DIR)


if __name__ == '__main__':
    main()
