import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from django.conf import settings


def unwrap_series_cells(s: pd.Series) -> pd.Series:
    def fix(x):
        if isinstance(x, pd.Series):
            # jeśli długość 1 – weź wartość
            if len(x) == 1:
                return x.iloc[0]
            # w innym wypadku zamień na listę (Parquet obsłuży listy skalara)
            return x.tolist()
        return x

    out = s.map(fix)
    return out


def unwrap_df_cells(s: pd.Series) -> pd.Series:
    def fix(x):
        if isinstance(x, pd.DataFrame):
            # np. lista rekordów (dictów)
            return x.to_dict(orient="records")
        return x

    return s.map(fix)


def convert_pkl():
    for p in settings.RESULTS_BASE_DIR.glob('**/*.pkl'):
        print(p)

        # try:
        df = pickle.loads(p.read_bytes())
        # print(p.with_name(p.stem + '.parquet'))
        for col in df.columns:
            s = df[col]
            # Path -> string
            if s.map(lambda x: isinstance(x, Path)).any():
                df[col] = s.astype('string')
                continue
            # Series w komórkach -> scalar/list
            if s.map(lambda x: isinstance(x, pd.Series)).any():
                df[col] = unwrap_series_cells(s)
            # DataFrame -> list of dicts (lub JSON)
            if s.map(lambda x: isinstance(x, pd.DataFrame)).any():
                df[col] = s.map(lambda x: x.to_dict(orient="records") if isinstance(x, pd.DataFrame) else x)
            # set -> list
            if s.map(lambda x: isinstance(x, set)).any():
                df[col] = s.map(lambda x: list(x) if isinstance(x, set) else x)
            # ndarray -> list
            if s.map(lambda x: isinstance(x, np.ndarray)).any():
                df[col] = s.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            if col == 'pipeline':
                df.drop(columns=["pipeline"], inplace=True)
        df.to_parquet(p.with_name(p.stem + '.parquet'))
        p.unlink()
        # except ValueError:
        #     print('Skip')


def fix_parquet():
    for p in settings.RESULTS_BASE_DIR.glob('*/classify_ratio_clinical_lsi_bsi/**/results.parquet'):
        print(p)
        df = pd.read_parquet(p)
        s= df.iloc[0]
        print(df.dtypes)
        print(s)
        break


if __name__ == '__main__':
    fix_parquet()
