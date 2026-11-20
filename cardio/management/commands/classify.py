import logging
from functools import lru_cache

import optuna
from django.conf import settings
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardio.helpers.features_labels import get_features
from cardio.helpers.parallel import parallel
from ._base import Command as BaseCommand
from ...helpers.data import flatten_data
from ...models import Data, Split, Classif

_logger = logging.getLogger('classify')


class Command(BaseCommand):
    help = 'Classify features using ensemble models with Optuna optimization'

    def add_arguments(self, parser):
        parser.add_argument('--n-trials', type=int, default=None,
                            help='Number of trials for Optuna optimization')
        parser.add_argument('--label', type=str, default=None,
                            help='Specific label to use (filters LABELS_COLUMN)')
        parser.add_argument('--features', type=str, default=None,
                            help='Specific features set to use (filters FEATURES_COLUMNS)')
        parser.add_argument('--model', type=str, default=None,
                            help='Specific model to use')
        parser.add_argument('--seed', type=int, default=None,
                            help='Specific seed to use')
        parser.add_argument('--replace', action='store_true', default=False,
                            help='Replace existing results')
        parser.add_argument('--no-confirm', action='store_true', default=False,
                            help='No need confirmation of running')

    def handle(self, *args, **options):
        n_trials = options['n_trials']
        label = options['label']
        features = options['features']
        model = options['model']
        seed = options['seed']

        if n_trials is not None:
            settings.N_TRIALS = n_trials
        if label is not None:
            settings.LABELS_COLUMN = {k: v for k, v in settings.LABELS_COLUMN.items() if k == label}
        if features is not None:
            settings.FEATURES_COLUMNS = {k: v for k, v in settings.FEATURES_COLUMNS.items() if k == features}
        if model is not None:
            settings.MODELS = {model: settings.MODELS[model]}
        if seed is not None:
            settings.SEEDS = [seed]

        for dataset in settings.DATASETS:
            _logger.info(f'Dataset: {dataset}')
            self.choose_dataset(dataset)
            self.classify_dataset(dataset, **options)

    @lru_cache
    def _get_data(self, dataset: str, features: str, label: str) -> Data:
        try:
            data = Data.objects.get(dataset=dataset, features=features, label=label)
        except Data.DoesNotExist:
            df = get_features()
            (X, X_cols), y = flatten_data(df, settings.FEATURES_COLUMNS[features]), df[label]
            X = X[X_cols]
            data = Data.objects.create(dataset=dataset, features=features, label=label, X=X, y=y)
        return data

    @lru_cache
    def _get_split(self, data: Data, seed: int) -> Split:
        try:
            split = Split.objects.get(data=data, seed=seed)
        except Split.DoesNotExist:
            X_train, X_test, y_train, y_test = train_test_split(
                data.X, data.y, test_size=0.2, random_state=seed, stratify=data.y
            )
            split = Split.objects.create(data=data, seed=seed,
                                         X_train=X_train, X_test=X_test,
                                         y_train=y_train, y_test=y_test)
        return split

    def classify_dataset(self, dataset: str, replace: bool, no_confirm: bool, **kw):
        parameters = []
        cases = 0
        for label in settings.LABELS_COLUMN:
            for features in settings.FEATURES_COLUMNS:
                data = self._get_data(dataset, features, label)

                for seed in settings.SEEDS:
                    split = self._get_split(data, seed)

                    for model_type in settings.MODELS:
                        cases += 1
                        if not replace and Classif.objects.filter(split=split, model=model_type).exists():
                            continue
                        parameters.append((split.id, model_type))

        if not no_confirm:
            ans = 'show'
            while ans == 'show':
                ans = input(f'Run {len(parameters)} of {cases} cases? [y/N/show]: ').lower()
                if ans == 'y':
                    pass
                elif ans == 'show':
                    for split_id, model in parameters:
                        split = Split.objects.get(id=split_id)
                        print(f'features={split.data.features} '
                              f'label={split.data.label} '
                              f'seed={split.seed} model={model}')
                else:
                    print(self.style.ERROR('Aborted'))
                    return
        else:
            _logger.info(f'Run {len(parameters)} of {cases} cases')

        worker_results = parallel(worker, parameters)


def make_model(model_dict, trial_or_best):
    if isinstance(trial_or_best, dict):
        opt_params = trial_or_best
        if 'wrapper' in model_dict:
            for k in trial_or_best.copy():
                param, value = k.split('__')
                opt_params.setdefault(param, {})
                opt_params[param][value] = trial_or_best.pop(k)
    else:
        opt_params = model_dict['opt_params'](trial_or_best)

    params = model_dict['params']
    wrapped_params = {}

    if 'wrapper' in model_dict:
        params = params.copy()
        for param, wrap in model_dict['wrapper'].items():
            wrapped_params[param] = wrap(**params.pop(param, {}), **opt_params.pop(param, {}))

    return model_dict['class'](**opt_params, **params, **wrapped_params)


def optimize(split: Split, model_type: str, model_dict: dict):
    _logger.debug(f'Optimization: {model_type}')

    def objective(trial):
        if model_type in settings.ENSEMBLE_MODELS:
            estimators = []
            for m_name, m_params in settings.BASE_MODELS.items():
                estimators.append((m_name, make_model(m_params, trial)))
            model_dict['params']['estimators'] = estimators

        model = make_model(model_dict, trial)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=split.seed)
        scores = cross_val_score(pipeline, split.X_train, split.y_train, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=settings.N_TRIALS, timeout=settings.TIMEOUT)

    _logger.debug(f"Najlepsze parametry: {study.best_params}")
    _logger.debug(f"Najlepsza dokładność: {study.best_value:.4f}")

    return {
        'best_params': study.best_params,
        'best_value': study.best_value
    }


def test(split: Split, model_type: str, model_dict: dict, best_params: dict):
    _logger.debug(f'Testing: {model_type}')
    best_params = best_params.copy()

    if model_type in settings.ENSEMBLE_MODELS:
        estimators = []
        for m_name, m_params in settings.BASE_MODELS.items():
            final_params = {}
            final_params.update(m_params['params'])
            for bp in best_params.copy():
                if bp.startswith(m_name):
                    final_params[bp.replace(f'{m_name}__', '')] = best_params.pop(bp)
            estimators.append((m_name, m_params['class'](**final_params)))
        # replace estimators left in model_dict by the last Optuna trial with the best ones
        model_dict['params']['estimators'] = estimators

    best_params = {k.replace(f'{model_type}__', ''): v for k, v in best_params.items()}
    model = make_model(model_dict, best_params)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(split.X_train, split.y_train)

    y_proba = pipeline.predict_proba(split.X_test)[:, 1]

    return {
        'y_test': list(split.y_test),
        'y_proba': y_proba.tolist(),
    }


def worker(split_id: int, model_type: str):
    split = Split.objects.get(id=split_id)
    model_dict = settings.MODELS[model_type]
    study = optimize(split, model_type, model_dict)
    test_results = test(split, model_type, model_dict, study['best_params'])
    Classif.objects.update_or_create(
        split=split,
        model=model_type,
        defaults=dict(
            best_value=study['best_value'],
            best_params=study['best_params'],
            y_proba=test_results['y_proba'],
            y_test=test_results['y_test']
        )
    )
