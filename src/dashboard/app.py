"""Dashboard locale per confrontare i risultati di TempSAL su UEyes.

Legge i file prodotti da src/evaluate_ueyes.py (results/<run_name>/metrics.csv
+ results/<run_name>/predictions/*.png) -- uno per checkpoint valutato
(tipicamente "baseline" = TempSAL originale, "finetuned" = dopo l'adattamento
a UEyes) -- e mostra grafici interattivi, filtrabili per categoria di UI,
con export CSV e PNG (quest'ultimo dal menu della singola figura Plotly).

Avvio: streamlit run app.py  (dalla cartella src/dashboard/)
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPSAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.join(TEMPSAL_DIR, 'results')
DATA_DIR = os.path.join(TEMPSAL_DIR, 'data_ueyes')

TIME_SLICES = 5
AGG_METRICS = ['CC', 'KLDIV', 'NSS', 'SIM']

# Original image / GT / predictions all live at different native resolutions
# (see TODO.md, Step 1/2). For plain display we upscale the 256x256
# predictions up to each image's own native size instead of downscaling
# everything to 256 -- the model really did predict at 256x256, but shrinking
# the original screenshot and GT to match would throw away real detail
# (small UI text/icons become illegible), whereas upscaling an already-smooth
# saliency map loses very little. The interactive diff heatmap is capped
# separately (DIFF_MAX_DIM) since UEyes images go up to ~2048x2161 and a
# same-resolution Plotly heatmap with per-cell hover would be too heavy for
# the browser.
DIFF_MAX_DIM = 480

# Classic ColorBrewer-style diverging stops: blue (pred < gt) -> white (equal)
# -> red (pred > gt), same idea as Figure 3 in the TempSAL paper (there used
# to compare two consecutive temporal slices; here for prediction vs GT).
DIFF_COLORSCALE = [[0.0, "rgb(33,102,172)"], [0.5, "rgb(247,247,247)"], [1.0, "rgb(178,24,43)"]]

st.set_page_config(page_title="TempSAL su UEyes", layout="wide")


@st.cache_data
def load_runs():
    """Un DataFrame per ogni cartella results/<run_name>/ che ha un metrics.csv."""
    runs = {}
    if not os.path.isdir(RESULTS_DIR):
        return runs
    for run_name in sorted(os.listdir(RESULTS_DIR)):
        csv_path = os.path.join(RESULTS_DIR, run_name, 'metrics.csv')
        if os.path.isfile(csv_path):
            runs[run_name] = pd.read_csv(csv_path)
    return runs


@st.cache_data
def image_filename_map():
    """image_id -> nome file reale (le immagini UEyes mischiano .jpg/.png/.jpeg)."""
    img_dir = os.path.join(DATA_DIR, 'images', 'val')
    if not os.path.isdir(img_dir):
        return {}
    return {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if not f.startswith('.')}


def slice_cols(metric, run_df):
    return [f'{metric}_{t}' for t in range(TIME_SLICES) if f'{metric}_{t}' in run_df.columns]


def native_size(path):
    """(w, h) of the original screenshot -- the reference size everything
    else for this image gets displayed at."""
    with Image.open(path) as im:
        return im.size


def capped_size(size, max_dim=DIFF_MAX_DIM):
    """size scaled down to fit within max_dim on its longest side, same
    aspect ratio -- keeps the interactive diff heatmap responsive."""
    w, h = size
    if max(w, h) <= max_dim:
        return (w, h)
    scale = max_dim / max(w, h)
    return (max(1, round(w * scale)), max(1, round(h * scale)))


def load_rgb_resized(path, size):
    with Image.open(path) as im:
        if im.size == size:
            return im.convert('RGB')
        return im.convert('RGB').resize(size, Image.BILINEAR)


def load_gray_resized(path, size):
    """Grayscale array in [0,1], resized to the given (w, h) -- used both for
    display and for the pred-vs-GT difference maps below."""
    with Image.open(path) as im:
        im = im.convert('L')
        if im.size != size:
            im = im.resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def diff_heatmap_figure(pred_arr, gt_arr, title):
    """pred - gt as a diverging heatmap: red = modello prevede più attenzione
    del ground truth in quel punto, blu = ne prevede meno."""
    diff = pred_arr - gt_arr
    fig = go.Figure(data=go.Heatmap(
        z=diff, zmin=-1, zmax=1, colorscale=DIFF_COLORSCALE,
        colorbar=dict(title="pred − GT", tickvals=[-1, 0, 1], ticktext=["meno", "=", "più"]),
    ))
    fig.update_layout(
        title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, scaleanchor='y'),
        yaxis=dict(visible=False, autorange='reversed'),
    )
    return fig


runs = load_runs()

st.title("TempSAL su UEyes")
st.caption("Confronto tra il checkpoint originale e il modello adattato, sul validation set UEyes.")

if not runs:
    st.warning(
        "Nessun risultato trovato in `results/`. Genera almeno una valutazione con:\n\n"
        "```\ncd src\npython evaluate_ueyes.py --run_name baseline "
        "--model_path ./checkpoints/multilevel_tempsal.pt\n```"
    )
    st.stop()

# --- Sidebar: filtri ---
st.sidebar.header("Filtri")
selected_runs = st.sidebar.multiselect("Run da mostrare", list(runs.keys()), default=list(runs.keys()))

if not selected_runs:
    st.info("Seleziona almeno un run dalla barra laterale.")
    st.stop()

combined = pd.concat(
    [df.assign(run=name) for name, df in runs.items() if name in selected_runs],
    ignore_index=True,
)

all_categories = sorted(combined['category'].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Categoria UI", all_categories, default=all_categories)
combined = combined[combined['category'].isin(selected_categories)]

st.sidebar.markdown(f"**{combined['image_id'].nunique()}** immagini nel filtro corrente")
st.sidebar.download_button(
    "Scarica CSV filtrato",
    combined.to_csv(index=False).encode('utf-8'),
    file_name="tempsal_ueyes_metrics.csv",
    mime="text/csv",
)
st.sidebar.caption("Per esportare un singolo grafico in PNG: menu ⋯ in alto a destra del grafico → 'Download plot as png'.")

tab_agg, tab_slices, tab_qual, tab_delta = st.tabs(
    ["Metriche aggregate", "Metriche per slice temporale", "Confronto qualitativo", "Distribuzione differenze"]
)

# --- Tab 1: metriche aggregate ---
with tab_agg:
    st.subheader("Mappa aggregata (0-5s): CC, KLDIV, NSS, SIM")
    split_by_category = st.checkbox("Suddividi per categoria UI", value=False)

    group_cols = ['run', 'category'] if split_by_category else ['run']
    means = combined.groupby(group_cols)[AGG_METRICS].mean().reset_index()

    cols = st.columns(len(AGG_METRICS))
    for col, metric in zip(cols, AGG_METRICS):
        fig = px.bar(
            means, x='run', y=metric,
            color='category' if split_by_category else 'run',
            barmode='group', title=metric,
        )
        fig.update_layout(showlegend=split_by_category, height=360)
        col.plotly_chart(fig, width='stretch')

    st.dataframe(means, width='stretch')

# --- Tab 2: metriche per slice temporale ---
with tab_slices:
    st.subheader("Le 5 slice temporali (0-1s ... 4-5s): CC e KLDIV per intervallo")
    metric_choice = st.radio("Metrica", ["Vol_CC", "Vol_KLDIV"], horizontal=True)

    fig = go.Figure()
    for run_name in selected_runs:
        run_df = combined[combined['run'] == run_name]
        cols = slice_cols(metric_choice, run_df)
        if not cols:
            continue
        means_per_slice = [run_df[c].mean() for c in cols]
        fig.add_trace(go.Scatter(
            x=[f"{t}-{t+1}s" for t in range(len(cols))],
            y=means_per_slice,
            mode='lines+markers',
            name=run_name,
        ))
    fig.update_layout(
        xaxis_title="Intervallo di osservazione",
        yaxis_title=metric_choice.replace('_', ' '),
        height=450,
    )
    st.plotly_chart(fig, width='stretch')

# --- Tab 3: confronto qualitativo ---
with tab_qual:
    st.subheader("Confronto visivo: immagine, ground truth, predizioni")
    image_ids = sorted(combined['image_id'].unique().tolist())
    if not image_ids:
        st.info("Nessuna immagine nel filtro corrente.")
    else:
        chosen_id = st.selectbox("Immagine", image_ids)
        show_diff = st.checkbox("Mostra mappa differenza (predizione − ground truth)", value=True)
        if show_diff:
            st.caption("Nelle mappe differenza: 🔴 rosso = il modello prevede *più* attenzione del ground truth in quel punto, 🔵 blu = ne prevede *meno*.")
        filenames = image_filename_map()
        src_name = filenames.get(chosen_id)
        slots_per_run = 2 if show_diff else 1

        if not src_name:
            st.warning(f"Immagine originale non trovata per {chosen_id}.")
        else:
            src_path = os.path.join(DATA_DIR, 'images', 'val', src_name)
            disp_size = native_size(src_path)  # tutto il resto viene portato a QUESTA risoluzione
            diff_size = capped_size(disp_size)
            st.caption(
                f"Immagine originale e ground truth alla risoluzione nativa ({disp_size[0]}×{disp_size[1]}); "
                f"le predizioni (fatte dal modello a 256×256) sono ingrandite per il confronto. "
                + (f"La mappa differenza è invece limitata a {diff_size[0]}×{diff_size[1]} per restare interattiva."
                   if diff_size != disp_size else "")
            )

            st.markdown("**Mappa aggregata (0-5s)**")
            cols = st.columns(2 + len(selected_runs) * slots_per_run)
            cols[0].image(load_rgb_resized(src_path, disp_size), caption="Immagine originale", width='stretch')
            gt_path = os.path.join(DATA_DIR, 'maps', 'val', f'{chosen_id}.png')
            gt_arr = load_gray_resized(gt_path, disp_size) if os.path.isfile(gt_path) else None
            if gt_arr is not None:
                cols[1].image(gt_arr, caption="Ground truth", width='stretch', clamp=True)
            gt_arr_diff = load_gray_resized(gt_path, diff_size) if os.path.isfile(gt_path) else None
            for i, run_name in enumerate(selected_runs):
                base = 2 + i * slots_per_run
                pred_path = os.path.join(RESULTS_DIR, run_name, 'predictions', f'{chosen_id}_agg.png')
                if os.path.isfile(pred_path):
                    cols[base].image(load_gray_resized(pred_path, disp_size),
                                      caption=f"Predizione ({run_name})", width='stretch', clamp=True)
                    if show_diff and gt_arr_diff is not None:
                        pred_arr_diff = load_gray_resized(pred_path, diff_size)
                        cols[base + 1].plotly_chart(
                            diff_heatmap_figure(pred_arr_diff, gt_arr_diff, run_name), width='stretch'
                        )

            st.markdown("**5 slice temporali**")
            for t in range(TIME_SLICES):
                st.caption(f"{t}-{t+1}s")
                row_cols = st.columns(1 + len(selected_runs) * slots_per_run)
                gt_slice_path = os.path.join(DATA_DIR, 'saliency_volumes_5', 'val', f'{chosen_id}_{t}.png')
                gt_slice_arr = load_gray_resized(gt_slice_path, disp_size) if os.path.isfile(gt_slice_path) else None
                if gt_slice_arr is not None:
                    row_cols[0].image(gt_slice_arr, caption="Ground truth", width='stretch', clamp=True)
                gt_slice_arr_diff = load_gray_resized(gt_slice_path, diff_size) if os.path.isfile(gt_slice_path) else None
                for i, run_name in enumerate(selected_runs):
                    base = 1 + i * slots_per_run
                    pred_slice_path = os.path.join(RESULTS_DIR, run_name, 'predictions', f'{chosen_id}_slice{t}.png')
                    if os.path.isfile(pred_slice_path):
                        row_cols[base].image(load_gray_resized(pred_slice_path, disp_size),
                                              caption=run_name, width='stretch', clamp=True)
                        if show_diff and gt_slice_arr_diff is not None:
                            pred_slice_arr_diff = load_gray_resized(pred_slice_path, diff_size)
                            row_cols[base + 1].plotly_chart(
                                diff_heatmap_figure(pred_slice_arr_diff, gt_slice_arr_diff, run_name),
                                width='stretch'
                            )

# --- Tab 4: distribuzione differenze ---
with tab_delta:
    st.subheader("Differenza per immagine tra due run")
    if len(selected_runs) < 2:
        st.info("Seleziona almeno 2 run nella barra laterale per confrontarli.")
    else:
        c1, c2 = st.columns(2)
        run_a = c1.selectbox("Run A", selected_runs, index=0)
        run_b = c2.selectbox("Run B", selected_runs, index=min(1, len(selected_runs) - 1))
        metric = st.selectbox("Metrica", AGG_METRICS)

        pivot = combined[combined['run'].isin([run_a, run_b])].pivot_table(
            index='image_id', columns='run', values=metric
        )
        if run_a in pivot.columns and run_b in pivot.columns:
            pivot = pivot.dropna(subset=[run_a, run_b])
            pivot['delta'] = pivot[run_b] - pivot[run_a]
            fig = px.histogram(pivot, x='delta', nbins=30,
                                title=f"{metric}: {run_b} − {run_a} (per immagine)")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')
            st.caption(
                f"Media: {pivot['delta'].mean():+.4f} · "
                f"Migliorate: {(pivot['delta'] > 0).sum()} · "
                f"Peggiorate: {(pivot['delta'] < 0).sum()} · "
                f"Invariate: {(pivot['delta'] == 0).sum()}"
            )
            worst = pivot.sort_values('delta').head(5)
            st.markdown("**5 immagini più peggiorate** (utili da ispezionare nel tab Confronto qualitativo)")
            st.dataframe(worst, width='stretch')
