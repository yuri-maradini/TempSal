# TODO — Transfer learning di TempSAL su UEyes

Questo file traccia lo stato del lavoro sulla tesi: cosa è stato fatto sull'ambiente/codice di TempSAL, cosa abbiamo scoperto analizzando il dataset UEyes, e il piano dettagliato per il transfer learning del modello di temporal saliency su UEyes. Pensato come promemoria per la stesura della tesi (metodologia + eventuali problemi incontrati).

---

## 1. Stato attuale — cosa è stato fatto

### 1.1 Setup ambiente (fresh-start venv)

- Ricreato da zero `.venv` in `Tempsal/` con Python 3.13.7 (il vecchio venv aveva librerie installate in modo scorretto).
- Questa macchina **non ha GPU NVIDIA** (solo Intel UHD Graphics integrata) → installato `torch`/`torchvision`/`torchaudio` in build **CPU-only** più recenti (2.13.0 / 0.28.0 / 2.11.0), al posto dei pin originali `torch==1.8.0+cu111` di `src/requirements.txt` (build CUDA 11.1, non installabile su Python 3.13 e comunque inutile senza GPU CUDA).
- Installate le altre librerie di `src/requirements.txt` (wandb, pycocotools, ftfy, regex, tqdm, ipywidgets, seaborn, einops, clip-anytorch), con `kornia` aggiornato a `0.8.3` (il pin originale `0.5.10` era incompatibile con torch 2.x).
- Rimosso `libgl1-mesa-glx` dai requirements: è un pacchetto di sistema Linux (apt), non un pacchetto pip, quindi non aveva senso nel file.
- Aggiunte tre dipendenze usate dal codice ma **mancanti** dal `requirements.txt` originale: `opencv-python` (usato da `dataloader_clean.py`, `generate_volumes.py`, `loss.py`, `utils.py` come `cv2`), `scipy` (usato da `model.py`, `utils.py`), `ipykernel` (necessario per eseguire i notebook `.ipynb` da VSCode).
- `src/requirements.txt` aggiornato con le versioni esatte effettivamente installate e verificate compatibili (`pip check` pulito).
- Commit: `Fix portability issues` (model.py + requirements.txt).

**Nota pratica per VSCode**: quando si apre un notebook (`inference.ipynb`, `notebook.ipynb`), bisogna selezionare manualmente il kernel `.venv` di questo progetto (in alto a destra nel notebook → "Select Kernel" → Python Environments → `.venv`). Se non viene selezionato, il notebook usa il Python globale di sistema e le import falliscono (es. `ModuleNotFoundError: No module named 'einops'`) anche se tutto è installato correttamente nel venv.

### 1.2 Bug di portabilità corretti nel codice

Il repo TempSAL originale è stato scritto assumendo sempre una macchina con GPU NVIDIA disponibile; su questa macchina (CPU-only) alcuni pezzi non funzionavano out-of-the-box:

- **`src/model.py`**, classe `PNASBoostedModelMultiLevel.__init__` (righe ~249, 250, 268, 275): sostituite le chiamate `.cuda()` fisse con `.to(device)`, e aggiunto `map_location=device` al `torch.load()` del checkpoint. Senza questa modifica, il caricamento del modello falliva con `AssertionError: Torch not compiled with CUDA enabled` su qualunque macchina senza GPU CUDA.
- **`src/dataloader.py`** (nuovo file, generato da Copilot durante i primi test, poi verificato e commitato): `train.py` importa `from dataloader import SaliconDataset`, ma il repo originale contiene solo `dataloader_clean.py` (che definisce `SaliconDataset`) — non un modulo chiamato `dataloader`. Creato `dataloader.py` come semplice re-export (`from dataloader_clean import SaliconDataset`) per far risolvere l'import. Nessuna logica duplicata: l'implementazione reale resta in `dataloader_clean.py`.
- Commit: `Fix missing dataloader module for train.py`.

**Bug noti ma NON ancora corretti** (non servivano per il test di inferenza, ma bloccheranno il training — vedi sezione 3):

- `src/train.py:81` — `from model import PNASBoostedModelMultilevel` (con "l" minuscola in "Multilevel"), ma la classe in `model.py` si chiama `PNASBoostedModelMultiLevel` (con "L" maiuscola). Import error se si prova a lanciare `train.py` con `--enc_model pnas_boosted_multi` così com'è.
- `.cuda()` fissi non ancora patchati (serviranno un fix analogo a quello di `model.py` se si vuole far girare/debuggare training o generazione volumi su questa macchina CPU-only): `src/loss.py` righe 95-96 (dentro `nss()`), `src/generate_volumes.py` riga 29 (`GaussianBlur2D().cuda()`), `src/train.py` riga 103 (`loss_func`). Su una macchina con GPU per il training vero e proprio questo non è un problema.

### 1.3 Test di inferenza

- Eseguito `src/inference.ipynb` con il checkpoint pre-addestrato `src/checkpoints/multilevel_tempsal.pt` su un'immagine di test SALICON (`src/testing/images/COCO_val2014_000000000192.jpg`).
- Output: mappa di salienza finale (0-5s) + 5 slice di salienza temporale (0-1s, 1-2s, ..., 4-5s), visivamente coerenti (la salienza si concentra su battitore/catcher/arbitro nella scena di baseball, dove ci si aspetta l'attenzione visiva).
- Conferma che l'intera pipeline (ambiente + modello + checkpoint) funziona correttamente end-to-end su CPU.

### 1.4 Analisi dataset UEyes

Dataset trovato in `UEyes_dataset/` (root del progetto tesi, fuori da `Tempsal/`). Struttura verificata:

```
UEyes_dataset/
├── README.md
├── image_types.csv              # Image Name;Category;Block;Train/Test (separatore ";")
├── images/                      # 1980 screenshot UI (webpage, desktop, mobile, poster), jpg/png
├── eyetracker_logs/             # 554 CSV grezzi Gazepoint (uno per partecipante × blocco)
├── saliency_maps/
│   ├── fixmaps_{1s,3s,7s}/      # 1980 file ciascuna
│   ├── heatmaps_{1s,3s,7s}/     # 1980 file ciascuna
│   └── overlay_heatmaps_{1s,3s,7s}/
└── scanpaths/
    └── paths_{1s,3s,7s}/<image_id>/<partecipante>.png   # 1980 cartelle ciascuna
```

**Interpretazione dei nomi file in `eyetracker_logs/`** (verificato con script di parsing su tutta la cartella): pattern `{blocco}_{partecipante}_fixations.csv`, es. `00_kh000_fixations.csv`.
- `{blocco}`: numero a 2 cifre, va da `00` a `55` **ma manca il 51** → 55 blocchi totali, non 56. Corrisponde alla colonna `Block` di `image_types.csv`. Ogni blocco raggruppa esattamente 36 immagini (55 × 36 = 1980, torna col totale immagini del dataset). Un file = un partecipante che ha visto tutte le 36 immagini del suo blocco (verificato contando i `MEDIA_NAME` unici in un file campione: 36).
- `{partecipante}`: prefisso `kh`/`KH` (**case inconsistente tra i file**, serve confronto case-insensitive) + id a 3 cifre. 62 partecipanti totali (combacia col README). Il numero di partecipanti per blocco **non è fisso** (varia 4-25 nei blocchi controllati) → non tutti i 62 partecipanti hanno visto tutti i 55 blocchi (design a rotazione).
- **File spuri da filtrare quando si itera sulla cartella**: `.DS_Store` (metadata macOS) e quattro `.~lock.<nome>.csv#` (lock file di LibreOffice/OpenOffice Calc, segno che quei CSV sono stati aperti in Calc durante la preparazione del dataset) — non sono CSV di dati, vanno esclusi dal parsing (es. filtrando per `endswith('_fixations.csv')`).
- **Attenzione — colonna `Block` di `image_types.csv` parzialmente corrotta**: alcuni valori sono in notazione scientifica stile Excel con virgola decimale (es. `1,00E+01` invece di `10`), altri restano numeri semplici (`"50"`). Probabile riapertura/risalvataggio in Excel con locale europeo che ha "corretto" solo alcune righe. Normalizzare prima di ogni confronto/join con: `int(float(str(v).replace(',', '.')))`.

**Verifica timestamp — ESITO POSITIVO**: il dataset contiene dati temporali reali, non solo heatmap statiche aggregate. Due livelli di granularità disponibili:

1. **Dati grezzi con timestamp per-fissazione** (`eyetracker_logs/*.csv`): formato nativo Gazepoint eye-tracker (v2.0, [spec API qui](https://www.gazept.com/dl/Gazepoint_API_v2.0.pdf)). Colonne rilevanti per la ricostruzione temporale: `MEDIA_NAME` (immagine osservata), `FPOGID` (id progressivo della fissazione), `FPOGS` (fixation start time, secondi), `FPOGD` (fixation duration, secondi), `FPOGV` (flag di validità), `FPOGX`/`FPOGY` (coordinate normalizzate 0-1 del punto di fissazione). Questi sono timestamp **misurati direttamente** dall'eye-tracker — a differenza di SALICON (usato da TempSAL), dove i timestamp per-fissazione non esistono nei file `.mat` originali e devono essere **stimati euristicamente** (vedi `utils.py::parse_fixations`, che fa matching tra fissazioni e campioni di mouse-tracking continuo via distanza spazio-temporale pesata). Su UEyes questo passaggio euristico non serve: i timestamp sono già lì.

   **Verifica empirica del formato (fatta caricando i CSV con pandas)**: ogni riga del CSV è già una **fissazione discreta**, non un campione grezzo del sensore — `FPOGID` non si ripete mai all'interno del blocco di righe di una stessa immagine (0 duplicati controllati su più immagini/partecipanti). Conferma indiretta: la durata media delle fissazioni nel file è ~297ms (std 134ms), coerente con la letteratura sulle fissazioni oculari (200-400ms) — se fossero raw samples a piena frequenza (60-150Hz) le durate/intervalli tra righe sarebbero dell'ordine dei 10-20ms. **Quindi UEyes non fornisce il flusso continuo di raw gaze points, solo la sequenza di fissazioni già rilevate dall'algoritmo real-time del Gazepoint** — comunque sufficiente e anzi ideale per costruire bin temporali custom (vedi sotto).

   Altra cosa verificata: `FPOGS` si azzera a ogni nuovo `MEDIA_NAME` (è già relativo all'onset dello stimolo, non un clock di sessione continuo), e l'ultima fissazione di ogni immagine termina sistematicamente intorno a ~7.0s (media 6.99s, std 0.065s su 36 immagini controllate) — conferma che ogni immagine è stata osservata per un tempo fisso di ~7s, coerente con il bucket massimo `_7s`.

2. **Mappe pre-aggregate multi-durata** (`saliency_maps/*_{1s,3s,7s}/`): fixmap/heatmap già calcolate a 3 finestre di osservazione **cumulative**: 0-1s, 0-3s, 0-7s. Coprono tutte le 1980 immagini.

**Differenza di schema rispetto a TempSAL — da tenere presente**: TempSAL usa 5 bin temporali **disgiunti** da 1 secondo l'uno (0-1s, 1-2s, 2-3s, 3-4s, 4-5s — finestra totale di osservazione 5s, vedi `utils.py`: `TIMESPAN = 5000`, `time_slices = 5`). UEyes fornisce invece 3 finestre **cumulative** (non disgiunte) fino a un massimo di 7s. Le due rappresentazioni non sono direttamente intercambiabili: bisogna scegliere come conciliarle (vedi Step 2 del piano sotto).

**Puoi restringerti a un intervallo custom (es. 1-5s)?** Sì. Dato che ogni fissazione ha `FPOGS` (inizio, già relativo all'onset dell'immagine) e `FPOGD` (durata), basta filtrare per `MEDIA_NAME` e sull'intervallo `[FPOGS, FPOGS+FPOGD]` per ottenere le fissazioni in qualunque finestra scelta — più facile e più preciso di quanto TempSAL abbia dovuto fare su SALICON (dove i timestamp erano stimati euristicamente, non misurati). L'unica scelta metodologica è come gestire una fissazione che scavalca un confine di bin (es. inizia a 0.8s, dura 0.5s, termina a 1.3s): assegnarla per intero al bin che contiene `FPOGS` (semplice, stesso criterio usato da TempSAL su SALICON — vedi `generate_volumes.py` riga 36) oppure spezzarla/pesarla proporzionalmente tra i bin coinvolti (più accurato, più complesso).

---

## 2. Piano transfer learning — step dettagliati

L'obiettivo è adattare il modello TempSAL (pre-addestrato su SALICON) alla distribuzione di UEyes (screenshot di interfacce utente, molto diversi dalle foto naturali di SALICON), sfruttando i dati temporali reali disponibili in `eyetracker_logs/`.

### Step 1 — Preparazione struttura dati base ✅ FATTO

Convertita UEyes nella struttura di cartelle attesa da `train.py`/`SaliconDataset`:

```
Tempsal/data_ueyes/
├── images/{train,val}/          # 1872 train / 108 val, screenshot UEyes
├── maps/{train,val}/            # heatmap finale aggregata 0-7s (target per pred_map)
└── fixation_maps/{train,val}/   # fixation map binaria 256x256 (target per NSS loss)
```

Generata da due nuovi script (riusabili anche per lo Step 2):
- **`src/ueyes_utils.py`**: `load_all_fixations()` legge e concatena tutti i 554 log di `eyetracker_logs/*.csv` (filtrando i file spuri `.DS_Store`/`.~lock.*`), tenendo solo le fissazioni valide (`FPOGV==1`). `normalize_block()` per la colonna `Block` corrotta di `image_types.csv`.
- **`src/prepare_ueyes.py`**: script principale, orchestratore di tutto lo Step 1.

**Decisioni prese** (con conferma dell'utente):
1. **Estensione immagini** (65% del dataset non è `.jpg`): invece di riconvertire tutto a `.jpg` (lossy sulle 1283 PNG), modificato `dataloader_clean.py::SaliconDataset` perché rilevi l'estensione reale di ogni immagine (`self.img_filenames` costruito in `__init__` da `os.listdir(img_dir)`) invece di assumerla fissa. `images/` viene copiata con `shutil.copy` (byte-per-byte, nessuna ricompressione).
2. **Fixation map finale**: NON usata `fixmaps_7s` di UEyes (corrotta da artefatti JPEG per il 35% delle immagini — vedi analisi in sezione 1.4). Ricostruita da zero dalle coordinate grezze in `eyetracker_logs/` (via `ueyes_utils.load_all_fixations`), genuinamente binaria per tutte le 1980 immagini.
3. **`maps/`**: usata `heatmaps_7s` di UEyes così com'è (è una mappa continua/sfumata, la fonte JPEG lossy per il 35% delle immagini non è un problema significativo qui), ma **ri-salvata come `.png`** invece di mantenere l'estensione originale mista — per uniformità con `fixation_maps/` (`SaliconDataset` usa un unico parametro `exten` condiviso da entrambe le cartelle).

**Bug trovato e corretto durante l'implementazione** (non previsto nel piano iniziale): `SaliconDataset.__getitem__` ridimensiona `gt` a 256×256 (`cv2.resize`) ma **non ridimensiona mai `fixations`** — funzionava per SALICON solo perché lì tutti i file di fixation map condividono già una risoluzione nativa fissa (640×480, vedi `utils.py` `W`/`H`). Le immagini UEyes hanno invece risoluzioni native diverse tra loro, quindi il batching (`torch.stack` dentro il `DataLoader`) falliva con un errore di shape mismatch. Soluzione: `prepare_ueyes.py` rasterizza le fixation map **direttamente a 256×256** (stessa risoluzione a cui finiscono comunque immagine e heatmap), invece che alla risoluzione nativa di ogni immagine — evita anche il rischio di perdere punti isolati in un downsampling successivo di una mappa già sparsa.

**Verifica finale**: testato con un vero `SaliconDataset` + `DataLoader` su `data_ueyes/` — caricamento di singoli sample (spanning sia sorgenti jpg che png) e di un batch completo (shape risultante `[8, 3, 256, 256]` immagini, `[8, 256, 256]` gt e fixations), nessun errore.

`data_ueyes/` è in `.gitignore` (come `data/` per SALICON) — non va committata.

**Punto lasciato aperto** (non risolto in questo step, era già nella checklist): aspect ratio. `img_transform` fa `transforms.Resize((256, 256))`, cioè uno **stretch forzato a quadrato**, applicato sia da PIL sull'immagine sia dal nostro `prepare_ueyes.py` sulla rasterizzazione delle fixation map (coordinate normalizzate mappate direttamente su una griglia 256×256 quadrata). Le foto naturali di SALICON tollerano questa distorsione; gli screenshot UI di UEyes (spesso molto più larghi che alti) potrebbero risentirne di più — da tenere a mente se i risultati fossero sotto le aspettative.

### Step 2 — Generare i volumi di salienza temporale (GT) per UEyes ✅ FATTO

Scelta **Opzione A** (bin disgiunti da 1s, `time_slices=5`, fedele a TempSAL — per un confronto diretto con il paper originale). Nuovo script: **`src/generate_volumes_ueyes.py`**, equivalente di `src/generate_volumes.py` ma per UEyes, riusando `ueyes_utils.load_all_fixations()` già scritto per lo Step 1.

**Output generato**:
```
data_ueyes/
├── fixation_volumes_5/{train,val}/{image_id}_{0..4}.png   # 9360 train + 540 val
└── saliency_volumes_5/{train,val}/{image_id}_{0..4}.png   # 9360 train + 540 val
```

**Regole di binning** (decise con l'utente):
- Ogni fissazione assegnata al bin `min(int(FPOGS), 4)` — usa solo l'istante di inizio, mai la durata, esattamente il criterio di `generate_volumes.py` originale (riga 36). Per le fissazioni a cavallo di un confine di bin, questo significa assegnazione per intero al bin che contiene `FPOGS` — scelta fatta per fedeltà metodologica con TempSAL (è la stessa identica regola), non per un limite tecnico.
- Le fissazioni oltre i 5s (UEyes traccia fino a ~7s, TempSAL/SALICON si ferma a `TIMESPAN=5000ms`) **non vengono scartate**: il `min(..., 4)` le clippa nell'ultimo bin, esattamente come fa la formula originale (`min(int(ts*time_slices/TIMESPAN), time_slices-1)`) con la propria coda oltre i 5s. Nessuna perdita di dati, coerenza totale con l'originale.

**Bug trovato nel codice di riferimento** (bloccante, non solo di portabilità): `utils.py::GaussianBlur2D`/`GaussianBlur1D` chiamano `F.conv1d` su un tensore 5D — `conv1d` richiede input 2D/3D, non l'ha mai supportato in nessuna versione di PyTorch. Verificato empiricamente (`RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 5, 480, 640]`), indipendentemente da CPU/GPU. È quindi molto probabile che i file già inclusi nel repo (`data/fixation_volumes_5-original/`) NON siano stati generati eseguendo letteralmente `generate_volumes.py` come pubblicato. Sostituito con `cv2.GaussianBlur`, stessi identici parametri dell'originale (sigma=25, kernel 201×201) — stesso effetto matematico, implementazione funzionante.

**Altri dettagli di fedeltà preservati**:
- Canvas fisso 640×480 (le stesse costanti `W`/`H` di `utils.py`), non la risoluzione nativa di ogni screenshot — stessa risoluzione intermedia che TempSAL usa per SALICON prima del resize a 256×256 in fase di training. Il downscale a 256×256 è lasciato al loader dello Step 3 (**da non ripetere l'errore trovato nello Step 1**: il loader dovrà ridimensionare esplicitamente anche questi volumi, non solo `gt`).
- Normalizzazione per un **unico massimo globale su tutto il volume a 5 canali** (non per-canale), replicando esattamente `utils.py::get_saliency_volume` (`saliency_volume / saliency_volume.max()`) — preserva le differenze di intensità relativa tra slice (uno slice con più fissazioni resta visibilmente più "acceso" degli altri dopo la normalizzazione).

**Verifica fatta**: conteggio file (9360+540 per entrambe le cartelle, combacia con 1872/108 × 5 slice), ispezione visiva delle 5 slice di un'immagine campione (pattern spaziali distinti e plausibili slice per slice, non identici né rumore casuale), controllo numerico (fissazioni totali per immagine sommate sulle 5 slice ≈ conteggio trovato nello Step 1, scarto di 1 dovuto alla differente risoluzione di rasterizzazione 640×480 vs 256×256 — non un bug).

### Step 3 — Correggere i bug del training script

- Fix del typo in `src/train.py:81`: `PNASBoostedModelMultilevel` → `PNASBoostedModelMultiLevel`.
- **Problema più importante**: così com'è, `train.py` non allena affatto il ramo temporale. `SaliconDataset` restituisce solo `(img, gt, fixations)` — nessun volume temporale. `loss_func`/`train()`/`validate()` calcolano la loss solo su `pred_map` (output finale aggregato), mai su `vol_pred` (output temporale, 5 canali). Inoltre, dentro `PNASBoostedModelMultiLevel.__init__` (`model.py`), sia `self.pnas_vol` che `self.pnas_sal` vengono **congelati esplicitamente** (`param.requires_grad = False` per entrambi) — solo i layer di "mixing"/boosting (`deconv_layer1..4`, `deconv_mix`) restano allenabili. Quindi il training script, allo stato attuale, può solo fare fine-tuning dei layer di combinazione finale usando la mappa aggregata come target — il ramo di salienza temporale resterebbe quello pre-addestrato su SALICON, mai adattato a UEyes.
- Per allenare davvero la salienza temporale su UEyes servono modifiche mirate:
  - Estendere `SaliconDataset` (o creare una sottoclasse, es. `SaliconVolDataset`) perché carichi e restituisca anche il volume temporale GT (i file generati nello Step 2).
  - Aggiungere in `train.py` un termine di loss che confronti `vol_pred` (shape `(batch, time_slices, H, W)`) con il volume GT, riusando `cc`/`kldiv` di `loss.py` — queste funzioni operano su tensori `(batch, H, W)`, quindi vanno applicate in loop su ciascuna delle `time_slices` slice (o con un reshape che tratti le slice come dimensione di batch aggiuntiva).
  - Decidere se scongelare `self.pnas_vol` in `model.py` (necessario se si vuole che il ramo temporale si adatti realmente a UEyes) oppure lasciarlo congelato e allenare solo i layer di mixing (transfer learning più "leggero", meno rischio di overfitting, ma il ramo temporale resta quello di SALICON).

### Step 4 — Strategia di fine-tuning

- UEyes ha 1980 immagini totali, molte meno delle ~10000 di SALICON usate per il pre-training originale → rischio concreto di overfitting con un fine-tuning completo del backbone.
- Raccomandazione: **congelare il backbone PNAS** (`train_enc=False`, come già previsto dal parametro esistente in `train.py`) e allenare solo: il ramo temporale scongelato (`pnas_vol`, se si segue la strada "temporale vero" descritta sopra) + i layer di mixing (`deconv_layer1..4`, `deconv_mix`, già allenabili di default).
- Learning rate basso, coerente con un fine-tuning e non un training da zero (es. `1e-5`/`1e-6` — il default di `train.py` è già `1e-5`).
- Poche epoch vista la dimensione del dataset; monitorare overfitting confrontando train/val loss (già loggato su `wandb`).
- Warm-start obbligatorio dal checkpoint esistente: `--model_path`/`--model_vol_path` puntati a `src/checkpoints/multilevel_tempsal.pt`.

### Step 5 — Validazione e metriche

- `validate()` in `train.py` calcola già CC, KLDIV, NSS, SIM — ma solo sulla mappa finale aggregata. Se si vuole anche una valutazione quantitativa della qualità della salienza *temporale* (non solo finale), va esteso analogamente allo Step 3: calcolare le stesse metriche per-slice tra `vol_pred` e il volume GT, e loggarle separatamente (utile anche per la tesi: permette di mostrare se il modello migliora slice per slice dopo il fine-tuning, non solo sulla mappa aggregata).
- Tenere da parte alcune predizioni + heatmap GT per un confronto visivo qualitativo (come fatto nel test di inferenza, sezione 1.3) da includere nella tesi.

### Step 6 — Housekeeping tecnico prima di lanciare il training vero

- Patchare i restanti `.cuda()` fissi (`loss.py:95-96`, `generate_volumes.py:29`, `train.py:103`) con lo stesso pattern `.to(device)` già usato in `model.py`, **solo se** si prevede di eseguire/debuggare anche solo in parte il training su questa macchina CPU-only. Se il training vero e proprio girerà su una macchina con GPU (es. Colab, cluster universitario), questo passaggio non è strettamente necessario ma resta comunque una buona idea per poter testare la pipeline dati in locale prima di lanciare run costose altrove.

---

## 3. Decisioni aperte (da prendere prima di procedere)

- [x] ~~Estensione immagini mista e qualità fixation map~~ — risolto nello Step 1: loader esteso per estensioni miste, fixation map ricostruita da `eyetracker_logs/` invece che da `fixmaps_7s`.
- [x] ~~Opzione A vs Opzione B per il volume di salienza temporale~~ — risolto nello Step 2: scelta l'Opzione A (bin disgiunti da 1s, fedeltà a TempSAL). Fissazioni oltre i 5s clippate nell'ultimo bin (non scartate), fissazioni a cavallo di un bin assegnate per intero al bin di `FPOGS`.
- [ ] Scongelare `pnas_vol` (fine-tuning "vero" del ramo temporale) o lasciarlo congelato e allenare solo i layer di mixing (transfer learning più conservativo).
- [ ] Resize con stretch (comportamento attuale) o con padding, per gli screenshot UI non quadrati.
- [ ] Dove eseguire il training vero (serve una GPU: Colab, cluster universitario, altra macchina disponibile).

---

## 4. File e riferimenti utili

- Codice modello: `Tempsal/src/model.py`
- Training script (da correggere/estendere): `Tempsal/src/train.py`
- Generazione volumi SALICON (da cui prendere ispirazione per UEyes): `Tempsal/src/generate_volumes.py`
- Utility (Gaussian blur, parsing fissazioni, loss helpers): `Tempsal/src/utils.py`
- Loss functions: `Tempsal/src/loss.py`
- Dataset loader: `Tempsal/src/dataloader_clean.py` (+ shim `Tempsal/src/dataloader.py`)
- Checkpoint pre-addestrato: `Tempsal/src/checkpoints/multilevel_tempsal.pt`
- Notebook di inferenza (funzionante, testato su CPU): `Tempsal/src/inference.ipynb`
- Dataset UEyes: `UEyes_dataset/` (root progetto tesi)
- Paper TempSAL: `TempSAL_2301.02315v2.pdf` (root progetto tesi)
- Paper UEyes: `UEyes_2402.05202v1.pdf` (root progetto tesi)
