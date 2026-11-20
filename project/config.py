import logging
import re
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

np.random.seed(42)

optuna.logging.set_verbosity(logging.WARNING)

BACKGROUND = 0

ORIG_LUNGS = 1
ORIG_HEART = 2
ORIG_BORDER = 3

RIGHT_LUNG = 1
LEFT_LUNG = 2
HEART = 3
BORDER = 4

HAS_BORDER = False

MARGIN = 10

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_BASE_DIR = BASE_DIR / '../results'
DATASET_NAME = None
RESULTS_DIR = None
SAVE_IMAGES = True
DEBUG_LEVEL = 1
SEG_DIR = Path('/invalid')
IMG_DIR = Path('/invalid')
DATA_DIR = BASE_DIR / '../data'

CURVE_ANGLES_COUNT = 7

DATASETS = ['seg04', 'seg04.1']

CLINICAL = {
    'wiek': 'Wiek',
    'niewydolnosc_serca': 'Niewydolność serca',
    'choroba_niedokrwienna_serca': 'Choroba niedokrwienna serca',
    'cukrzyca': 'Cukrzyca 2',
    'nadcisnienie_tetnicze': 'Nadciśnienie tętnicze',
    'zaburzenia_lipidowe': 'Zaburzenia lipidowe',
    'af': 'AF',
    'plec': 'Plec 1-female, 0-male',
}

FEATURES_COLUMNS = {
    'bsi': ['u_area', 'c_area', 'l_area', 'r_area'],
    'lsi': ['slope', 'angle', 'left_radius'],
    'curve': ['curve_radius', 'curve_angles'],
    'bsi_norm': ['norm_u_area', 'norm_c_area', 'norm_l_area', 'norm_r_area'],
    'ratio': ['heart_lungs_ratio'],
    'clinical': list(CLINICAL),
    'clinical2': ['nadcisnienie_tetnicze', 'niewydolnosc_serca'],
    'heart_grid_6x6': ['heart_grid_6x6'],
    'heart_widths_50': ['heart_widths_50'],
    'left_lung_widths_50': ['left_lung_widths_50'],
    'heart_widths_50_30-40': [f'heart_widths_50_{i}' for i in range(30, 41)],
}
FEATURES_COLUMNS['bsi_lsi'] = FEATURES_COLUMNS['bsi'] + FEATURES_COLUMNS['lsi']
FEATURES_COLUMNS['bsi_lsi_curve'] = FEATURES_COLUMNS['bsi'] + FEATURES_COLUMNS['lsi'] + FEATURES_COLUMNS['curve']
FEATURES_COLUMNS['ratio_clinical'] = FEATURES_COLUMNS['ratio'] + FEATURES_COLUMNS['clinical']
FEATURES_COLUMNS['ratio_clinical2'] = FEATURES_COLUMNS['ratio'] + FEATURES_COLUMNS['clinical2']
FEATURES_COLUMNS['ratio_clinical_lsi_bsi'] = FEATURES_COLUMNS['ratio'] + FEATURES_COLUMNS['clinical'] + \
                                             FEATURES_COLUMNS['bsi'] + FEATURES_COLUMNS['lsi']
FEATURES_COLUMNS['ratio_clinical_bsi'] = FEATURES_COLUMNS['ratio'] + FEATURES_COLUMNS['clinical'] + \
                                         FEATURES_COLUMNS['bsi']
FEATURES_COLUMNS['heart_widths_50_clinical2'] = FEATURES_COLUMNS['heart_widths_50'] + FEATURES_COLUMNS['clinical2']
FEATURES_COLUMNS['heart_widths_50_30-40_clinical2'] = FEATURES_COLUMNS['heart_widths_50_30-40'] + \
                                                      FEATURES_COLUMNS['clinical2']

LABELS_FILE = BASE_DIR / '../database.xlsx'
LABELS_COLUMN = {
    'label_or': 'Cardiomegaly (Hogher LVEDV/BSA lub LVM/BSA)',
    'label_lvedv': 'Higher LVEDV/BSA',
    'label_lvm': 'Higher LVM/BSA'
}

BASE_MODELS = {
    'KNN': {
        'class': KNeighborsClassifier,
        'opt_params': lambda trial: {
            'n_neighbors': trial.suggest_int('KNN__n_neighbors', 1, 20)
        },
        'params': {}
    },
    'SVM': {
        'class': SVC,
        'opt_params': lambda trial: {
            'C': trial.suggest_float('SVM__C', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('SVM__kernel', ['linear', 'rbf', 'poly']),
        },
        'params': {
            'probability': True,
            # libsvm has no iteration limit by default and can hang on degenerate folds;
            # normal fits on this data need up to ~2.2M iterations
            'max_iter': 10_000_000,
        }
    },
    'MLP': {
        'class': MLPClassifier,
        'opt_params': lambda trial: {
            'hidden_layer_sizes': trial.suggest_int('MLP__hidden_layer_sizes', 50, 200),
            'alpha': trial.suggest_float('MLP__alpha', 1e-5, 1e-1, log=True),
        },
        'params': {
            'max_iter': 5000,
            'random_state': 42
        }
    },
}

ENSEMBLE_MODELS = {
    'Voting': {
        'class': VotingClassifier,
        'opt_params': lambda trial: {},
        'params': {
            'voting': 'soft',
            'weights': [1, 1, 1],
        }
    },
    'Stacking': {
        'class': StackingClassifier,
        'wrapper': {
            'final_estimator': LogisticRegression
        },
        'opt_params': lambda trial: {
            'final_estimator': {
                'solver': trial.suggest_categorical('Stacking__final_estimator__solver', ['lbfgs', 'liblinear']),
            },
        },
        'params': {
            'passthrough': True,
            'cv': 5,
            'final_estimator': {
                'max_iter': 300
            },
        }
    },
}

MODELS = BASE_MODELS | ENSEMBLE_MODELS

SEEDS = np.random.choice(10000, size=50, replace=False).tolist()
N_TRIALS = 50
TIMEOUT = 900  # 15 minut
N_PROC = 70

DIR_NAMES = {

    'predictions_suwk_03_newest': 'newest',
    'predictions_suwk_03_oldest': 'oldest',
    'predictionsTs': 'first',
    'imagesTs': 'first',
    'suwk_03_newest': 'newest',
    'suwk_03_oldest': 'oldest',

    'test': 'newest',
}

NR_REGEXP = {
    'predictions_suwk_03_newest': re.compile(r'^ANON-PATIENT_NAME_(\d+).nii.gz$'),
    'predictions_suwk_03_oldest': re.compile(r'^ANON-PATIENT_NAME_(\d+).nii.gz$'),
    'predictionsTs': re.compile(r'^(\d+)[A-Za-z]+.nii.gz$'),
    'suwk_03_newest': re.compile(r'^ANON-PATIENT_NAME_(\d+)_0000.nii.gz$'),
    'suwk_03_oldest': re.compile(r'^ANON-PATIENT_NAME_(\d+)_0000.nii.gz$'),
    'imagesTs': re.compile(r'^(\d+)[A-Za-z]+_0000.nii.gz$'),

}
EXCLUDE = {
    'oldest_0043',
}

LOG_DIR = BASE_DIR / '../logs'
LOG_DIR.mkdir(exist_ok=True)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
        },
        'features': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_DIR / 'features.log',
            'maxBytes': 50 * 1024 * 1024,
            'backupCount': 10000,
            'formatter': 'default'
        },

    },
    'loggers': {
        'features': {
            'handlers': ['console', 'azai.api'],
            'level': 'INFO'
        },
    }
}
for name in ['features', 'classify', 'analyze']:
    LOGGING['handlers'][name] = {
        'level': 'DEBUG',
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': LOG_DIR / f'{name}.log',
        'maxBytes': 50 * 1024 * 1024,
        'backupCount': 10000,
        'formatter': 'default'
    }
    LOGGING['loggers'][name] = {
        'handlers': ['console', name],
        'level': 'INFO'
    }
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'cardio.sqlite3',
    }
}