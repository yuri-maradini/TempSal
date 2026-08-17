import os
import re

import pandas as pd

LOG_FILENAME_RE = re.compile(r'^\d+_[A-Za-z]+\d+_fixations\.csv$', re.IGNORECASE)


def list_log_files(logs_dir):
    """List valid Gazepoint fixation log files, skipping OS/editor cruft
    (.DS_Store, LibreOffice .~lock.*.csv# files) that live in the same folder.
    """
    return sorted(f for f in os.listdir(logs_dir) if LOG_FILENAME_RE.match(f))


def load_all_fixations(logs_dir, verbose=True):
    """Load and concatenate every participant fixation log in logs_dir.

    Returns a DataFrame with columns MEDIA_NAME, FPOGS, FPOGD, FPOGX, FPOGY
    containing only valid fixations (FPOGV == 1) from all participants/blocks.
    FPOGS/FPOGD are in seconds, relative to that image's onset. FPOGX/FPOGY
    are normalized (0-1) gaze coordinates.
    """
    files = list_log_files(logs_dir)
    cols = ['MEDIA_NAME', 'FPOGS', 'FPOGD', 'FPOGX', 'FPOGY', 'FPOGV']

    frames = []
    iterator = files
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(files, desc='Reading eyetracker logs')

    for f in iterator:
        df = pd.read_csv(os.path.join(logs_dir, f), usecols=cols)
        df = df[df['FPOGV'] == 1]
        frames.append(df)

    all_fixations = pd.concat(frames, ignore_index=True)
    return all_fixations.drop(columns=['FPOGV'])


def normalize_block(value):
    """UEyes' image_types.csv Block column is partly corrupted by an Excel
    resave (some values got turned into scientific notation with a comma
    decimal separator, e.g. "1,00E+01" instead of "10"). Normalize any of
    these representations back to a plain int.
    """
    return int(float(str(value).replace(',', '.')))
