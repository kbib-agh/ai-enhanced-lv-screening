from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management import call_command
from django.db.transaction import atomic
from sklearn.model_selection import train_test_split

from cardio.helpers.data import flatten_data
from cardio.management.commands._base import Command as BaseCommand
from cardio.models import Feature, Data, Split, Classif


class Command(BaseCommand):
    help = 'Import data to DB'

    datasets = ['seg03', 'seg03.1', 'seg04', 'seg04.1']

    def handle(self, *args, **options):
        self.make_features()
        self.import_datasets()
        self.import_results()

    @atomic
    def make_features(self):
        call_command('features')

    @atomic
    def import_datasets(self):
        for dataset in self.datasets:
            self.choose_dataset(dataset)
            df = Feature.df_objects.filter(dataset=dataset).to_dataframe()
            for features in settings.FEATURES_COLUMNS:
                for label in settings.LABELS_COLUMN:
                    print(dataset, features, label)

                    X, y = flatten_data(df[settings.FEATURES_COLUMNS[features]]), df[label]

                    data = Data.objects.create(
                        dataset=dataset,
                        features=features,
                        label=label,
                        X=X,
                        y=y,
                    )
                    for seed in settings.SEEDS:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=seed, stratify=y
                        )
                        split = Split.objects.create(data=data, seed=seed,
                                                     X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    @atomic
    def import_results(self):
        for dataset in self.datasets:
            self.choose_dataset(dataset)
            for features in settings.FEATURES_COLUMNS:
                for label in settings.LABELS_COLUMN:
                    print(dataset, features, label)
                    results_dir = (Path(str(settings.RESULTS_DIR).replace('/results/', '/results-2025.10.16/')) /
                                   f'classify_{features}' / label)
                    try:
                        results_df = pd.read_parquet(results_dir / 'results.parquet')
                    except FileNotFoundError as e:
                        print(self.style.ERROR(f'Could not find parquet {e.filename}'))
                        continue

                    # print(results_df.iloc[0])
                    data = Data.objects.get(dataset=dataset, features=features, label=label)
                    for idx, row in results_df.iterrows():
                        split = Split.objects.get(data=data, seed=row['seed'])

                        Classif.objects.create(
                            split=split,
                            best_params=row['best_params'],
                            best_value=row['best_value'],
                            y_test=row['y_test'],
                            y_proba=row['y_proba'],
                            model=row['model'],
                        )
