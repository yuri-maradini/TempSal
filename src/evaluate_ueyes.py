"""Evaluate a TempSAL checkpoint on the UEyes validation (or train) split:
computes per-image aggregate metrics (CC/KLDIV/NSS/SIM) and per-slice
metrics (CC/KLDIV for each of the 5 temporal slices), and saves the
predicted maps as PNGs.

Writes results/<run_name>/metrics.csv (one row per image) and
results/<run_name>/predictions/*.png -- this is the data the dashboard
(src/dashboard/app.py) reads. Run it once per checkpoint you want to
compare (e.g. --run_name baseline against multilevel_tempsal.pt, later
--run_name finetuned against the fine-tuned checkpoint); the dashboard
picks up however many runs/ subfolders exist.
"""
import argparse
import os

import pandas as pd
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import SaliconDataset
from loss import cc, kldiv, nss, similarity
from model import PNASBoostedModelMultiLevel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UEYES_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'UEyes_dataset')


def load_categories():
    df = pd.read_csv(os.path.join(UEYES_DIR, 'image_types.csv'), sep=';')
    df['ImageId'] = df['Image Name'].apply(lambda n: os.path.splitext(n)[0])
    return dict(zip(df['ImageId'], df['Category']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True, help='es. "baseline" o "finetuned" -> results/<run_name>/')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_vol_path', default=None, help='default: uguale a --model_path')
    parser.add_argument('--dataset_dir', default='../data_ueyes/')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--time_slices', default=5, type=int)
    parser.add_argument('--limit', default=None, type=int, help='valuta solo le prime N immagini (debug rapido)')
    args = parser.parse_args()

    model_vol_path = args.model_vol_path or args.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_dir = args.dataset_dir + f"images/{args.split}/"
    gt_dir = args.dataset_dir + f"maps/{args.split}/"
    fix_dir = args.dataset_dir + f"fixation_maps/{args.split}/"
    vol_dir = args.dataset_dir + f"saliency_volumes_5/{args.split}/"

    img_ids = sorted(nm.split(".")[0] for nm in os.listdir(img_dir))
    if args.limit:
        img_ids = img_ids[:args.limit]

    dataset = SaliconDataset(img_dir, gt_dir, fix_dir, img_ids, vol_dir=vol_dir, time_slices=args.time_slices)

    print(f"Carico il modello da {args.model_path} (device: {device}) ...")
    model = PNASBoostedModelMultiLevel(device, args.model_path, model_vol_path, args.time_slices,
                                        train_model=False, train_enc=False)
    model = model.to(device)
    model.eval()

    categories = load_categories()

    out_dir = os.path.join(SCRIPT_DIR, '..', 'results', args.run_name)
    pred_dir = os.path.join(out_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    rows = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Valutazione ({args.run_name}/{args.split})"):
            img_id = img_ids[i]
            img, gt, fixations, vol = dataset[i]
            img_b = img.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)
            fix_b = fixations.unsqueeze(0).to(device)
            vol_b = vol.unsqueeze(0).to(device)

            pred_map, vol_pred = model(img_b)

            row = {
                'image_id': img_id,
                'category': categories.get(img_id, 'unknown'),
                'split': args.split,
                'CC': cc(pred_map, gt_b).item(),
                'KLDIV': kldiv(pred_map, gt_b).item(),
                'NSS': nss(pred_map, fix_b).item(),
                'SIM': similarity(pred_map, gt_b).item(),
            }
            for t in range(args.time_slices):
                row[f'Vol_CC_{t}'] = cc(vol_pred[:, t], vol_b[:, t]).item()
                row[f'Vol_KLDIV_{t}'] = kldiv(vol_pred[:, t], vol_b[:, t]).item()
            rows.append(row)

            save_image(pred_map, os.path.join(pred_dir, f'{img_id}_agg.png'), normalize=True)
            for t in range(args.time_slices):
                save_image(vol_pred[:, t], os.path.join(pred_dir, f'{img_id}_slice{t}.png'), normalize=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSalvato {csv_path} ({len(df)} righe)")
    print(df[['CC', 'KLDIV', 'NSS', 'SIM']].mean())


if __name__ == '__main__':
    main()
