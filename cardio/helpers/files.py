from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def open_nii(path: Path, normalize: bool = False) -> np.ndarray:
    img = nib.load(path).get_fdata()
    if normalize:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = np.ascontiguousarray(np.round(np.squeeze(img)).astype(np.uint8))
    return img


def is_listlike(col):
    for val in col:
        if val is None:
            continue
        if isinstance(val, (list, np.ndarray, pd.DataFrame, pd.Series, Pipeline)):
            return True
        else:
            return False
    return False
