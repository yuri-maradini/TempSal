"""Quantifies and visualizes center-bias in predicted saliency maps.

Compares the baseline checkpoint (SALICON-only) against the UEyes
fine-tuned checkpoints: for each run's predicted aggregate maps
(results/<run>/predictions/*_agg.png, produced by evaluate_ueyes.py),
computes the intensity-weighted centroid of each map and its distance
from the image center, plus the dataset-average map (a standard way to
visualize center-bias: individual content cancels out, a systematic
positional bias does not).

Requires results/<run>/predictions/ to already exist for each run in
RUNS (run evaluate_ueyes.py first). Ground truth comes from
data_ueyes/maps/val/.

Writes results/presentation/center_bias_average_maps.png, prints per-run
and per-category statistics plus a summary table of centroid distances
(baseline / each fine-tuned run / ground truth), and saves that table
as results/presentation/centroid_distance_table.{csv,md}.
"""
import csv
import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
GT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_ueyes', 'maps', 'val')

RUNS = ['baseline', 'finetuned', 'finetuned_v2', 'finetuned_v3', 'finetuned_v4']
RUN_LABELS = {
    'baseline': 'Baseline (solo SALICON)',
    'finetuned': 'Fine-tuned v1',
    'finetuned_v2': 'Fine-tuned v2',
    'finetuned_v3': 'Fine-tuned v3',
    'finetuned_v4': 'Fine-tuned v4',
    'ground_truth': 'Ground truth (UEyes)',
}
SIZE = 256


def to_markdown_table(df):
    header = '| ' + ' | '.join(df.columns) + ' |'
    sep = '|' + '|'.join('---' for _ in df.columns) + '|'
    rows = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([header, sep] + rows)


def load_gray(path, resize=None):
    im = Image.open(path).convert('L')
    if resize is not None:
        im = im.resize(resize, Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def centroid(map_arr):
    s = map_arr.sum()
    if s <= 0:
        return 0.5, 0.5
    h, w = map_arr.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = (map_arr * xs).sum() / s / (w - 1)
    cy = (map_arr * ys).sum() / s / (h - 1)
    return cx, cy


def dist_from_center(cx, cy):
    return float(np.hypot(cx - 0.5, cy - 0.5))


def load_image_ids():
    csv_path = os.path.join(RESULTS_DIR, 'baseline', 'metrics.csv')
    image_ids, categories = [], {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            image_ids.append(row['image_id'])
            categories[row['image_id']] = row['category']
    return image_ids, categories


def run_stats(run, image_ids):
    dists, avg_map = [], np.zeros((SIZE, SIZE), dtype=np.float64)
    for iid in image_ids:
        p = os.path.join(RESULTS_DIR, run, 'predictions', f'{iid}_agg.png')
        m = load_gray(p)
        cx, cy = centroid(m)
        dists.append(dist_from_center(cx, cy))
        s = m.sum()
        if s > 0:
            avg_map += m / s
    return np.array(dists), avg_map / len(image_ids)


def gt_stats(image_ids):
    dists, avg_map = [], np.zeros((SIZE, SIZE), dtype=np.float64)
    for iid in image_ids:
        found = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(GT_DIR, iid + ext)
            if os.path.exists(cand):
                found = cand
                break
        m = load_gray(found, resize=(SIZE, SIZE))
        cx, cy = centroid(m)
        dists.append(dist_from_center(cx, cy))
        s = m.sum()
        if s > 0:
            avg_map += m / s
    return np.array(dists), avg_map / len(image_ids)


def main():
    image_ids, categories = load_image_ids()
    print(f'# images: {len(image_ids)}')

    dist_by_run, avgmap_by_run = {}, {}
    for run in RUNS:
        pred_dir = os.path.join(RESULTS_DIR, run, 'predictions')
        if not os.path.isdir(pred_dir):
            print(f'[skip] {run}: results/{run}/predictions non trovato (lanciare evaluate_ueyes.py --run_name {run} prima)')
            continue
        dists, avg_map = run_stats(run, image_ids)
        dist_by_run[run] = dists
        avgmap_by_run[run] = avg_map
        print(f'{run}: n={len(dists)} mean_dist_from_center={dists.mean():.4f} std={dists.std():.4f}')

    gt_dists, gt_avg_map = gt_stats(image_ids)
    print(f'ground_truth: n={len(gt_dists)} mean_dist_from_center={gt_dists.mean():.4f} std={gt_dists.std():.4f}')

    # summary table: baseline / each fine-tuned run / ground truth
    table_rows = []
    for run, dists in dist_by_run.items():
        table_rows.append((RUN_LABELS.get(run, run), len(dists), round(dists.mean(), 4), round(dists.std(), 4)))
    table_rows.append((RUN_LABELS['ground_truth'], len(gt_dists), round(gt_dists.mean(), 4), round(gt_dists.std(), 4)))
    table_df = pd.DataFrame(table_rows, columns=['Run', 'n', 'Distanza media dal centro', 'Std'])

    print('\nTabella riassuntiva — distanza dei centroidi dal centro:')
    print(table_df.to_string(index=False))

    presentation_dir = os.path.join(RESULTS_DIR, 'presentation')
    os.makedirs(presentation_dir, exist_ok=True)
    table_df.to_csv(os.path.join(presentation_dir, 'centroid_distance_table.csv'), index=False)
    with open(os.path.join(presentation_dir, 'centroid_distance_table.md'), 'w', encoding='utf-8') as f:
        f.write(to_markdown_table(table_df) + '\n')
    print(f'Salvato {os.path.join(presentation_dir, "centroid_distance_table.csv")}')
    print(f'Salvato {os.path.join(presentation_dir, "centroid_distance_table.md")}')

    if 'baseline' in dist_by_run:
        base = dist_by_run['baseline']
        print('\nConfronto vs baseline (positivo = fine-tuned meno centrato del baseline):')
        for run, dists in dist_by_run.items():
            if run == 'baseline':
                continue
            diff = base - dists
            print(f'  baseline vs {run}: mean(baseline-{run})={diff.mean():.4f}, '
                  f'immagini con {run} meno centrato (piu` lontano dal centro) del baseline: {(diff < 0).mean():.1%}')

        print('\nPer categoria (baseline vs ultimo run fine-tuned disponibile):')
        last_ft = [r for r in RUNS[1:] if r in dist_by_run]
        if last_ft:
            last_ft = last_ft[-1]
            for cat in sorted(set(categories.values())):
                idxs = [i for i, iid in enumerate(image_ids) if categories[iid] == cat]
                b = dist_by_run['baseline'][idxs].mean()
                f = dist_by_run[last_ft][idxs].mean()
                g = gt_dists[idxs].mean()
                print(f'  {cat}: baseline={b:.4f} {last_ft}={f:.4f} gt={g:.4f} (n={len(idxs)})')

    # figure: baseline vs best available fine-tuned run vs ground truth
    panels = [('baseline', 'Baseline (solo SALICON)')]
    for r in ('finetuned_v4', 'finetuned_v3', 'finetuned_v2', 'finetuned'):
        if r in avgmap_by_run:
            panels.append((r, f'{RUN_LABELS.get(r, r)} (UEyes)'))
            break
    panels.append(('gt', 'Ground truth (UEyes)'))

    maps = dict(avgmap_by_run)
    maps['gt'] = gt_avg_map
    vmax = max(maps[k].max() for k, _ in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
    for ax, (key, title) in zip(axes, panels):
        m = maps[key]
        ax.imshow(m, cmap='inferno', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(m.shape[1] / 2, m.shape[0] / 2, marker='+', color='cyan', markersize=12, markeredgewidth=2)
    fig.suptitle('Mappa di salienza media sulle immagini di validazione UEyes', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(presentation_dir, 'center_bias_average_maps.png')
    fig.savefig(out_path, dpi=150)
    print(f'\nSalvato {out_path}')


if __name__ == '__main__':
    main()
