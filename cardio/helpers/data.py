from itertools import chain
from typing import Iterable

import pandas as pd


def flatten_data(data: pd.DataFrame, columns: Iterable[str] = None) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    data_columns = data.columns.to_list()
    columns = set(columns) if columns else set()

    flat_cols = []
    for col_name in data_columns:
        if isinstance(data[col_name][0], list):
            col = data[col_name]
            while isinstance(col[0][0], list):
                col = col.apply(lambda x: list(chain.from_iterable(x)))
            # Spłaszczamy listy do kolumn
            expanded = pd.DataFrame(col.to_list(), index=data.index)
            expanded.columns = [f"{col_name}_{i}" for i in range(expanded.shape[1])]

            # Usuwamy kolumnę z kopii DataFrame i doklejamy rozwinięte kolumny
            data = pd.concat([data.drop(col_name, axis=1), expanded], axis=1)
            if col_name in columns:
                columns.remove(col_name)
                flat_cols.extend(expanded.columns.tolist())
    if columns:
        flat_cols.extend(columns)
        return data, flat_cols
    return data
