import pandas as pd

from django.conf import settings

from cardio.models import Feature


def get_label_map() -> dict[str, dict]:
    y_map = {}
    for sheet in settings.DIR_NAMES.values():
        labels_df = pd.read_excel(settings.LABELS_FILE)
        for idx, row in labels_df.iterrows():
            labels = {k: row[v] for k, v in settings.CLINICAL.items()}
            for label_key, label_column in settings.LABELS_COLUMN.items():
                labels[label_key] = row[label_column]
            y_map[row['ID']] = labels
    return y_map


def get_features():
    # features_df = pd.read_parquet(settings.RESULTS_DIR / 'features.parquet')
    features_df = Feature.df_objects.filter(dataset=settings.DATASET_NAME).to_dataframe()
    return features_df
