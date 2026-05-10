# prepare_propofol.py
import os
import numpy as np
import pandas as pd
from bids import BIDSLayout
from topoconscious.preprocessing import Preprocessor

def load_subject_data(bids_path, subject, session, atlas='schaefer100', tr=2.0):
    """Load and preprocess BOLD time series for a single subject & session."""
    layout = BIDSLayout(bids_path, validate=False)
    bold_file = layout.get(subject=subject, session=session, suffix='bold',
                           extension='.nii.gz', return_type='file')[0]
    preproc = Preprocessor(atlas=atlas, tr=tr)
    return preproc.load_and_extract(bold_file)

# === Configuration ===
BIDS_DIR = "./data/ds003171_test"
OUTPUT_DIR = "./validation_data_propofol"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all subjects present in the BIDS directory
layout = BIDSLayout(BIDS_DIR, validate=False)
subjects = layout.get_subjects()

ts_list = []
labels = []

for sub in subjects:
    # Get all sessions for this subject
    sessions = layout.get_sessions(subject=sub)
    for sess in sessions:
        # Determine label from session name
        if sess == "awake":
            label = 1
        elif sess == "deep_sedation":
            label = 0
        else:
            # Skip mild_sedation and recovery for binary classification
            continue
        
        try:
            ts = load_subject_data(BIDS_DIR, sub, sess)
            np.save(f"{OUTPUT_DIR}/{sub}_{sess}_ts.npy", ts)
            ts_list.append(ts)
            labels.append(label)
            print(f"✓ Processed {sub} ({sess}) -> label={label}")
        except Exception as e:
            print(f"✗ Failed {sub} ({sess}): {e}")

# Save the arrays and labels
np.save(f"{OUTPUT_DIR}/ts_list.npy", np.array(ts_list, dtype=object))
np.save(f"{OUTPUT_DIR}/labels.npy", np.array(labels))

print(f"\n Prepared {len(ts_list)} samples (awake={sum(labels)}, deep={len(labels)-sum(labels)})")
