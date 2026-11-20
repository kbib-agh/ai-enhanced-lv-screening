from django.conf import settings
from django.core.management.base import BaseCommand

from cardio.helpers.features_labels import get_features


class Command(BaseCommand):

    def choose_dataset(self, dataset: str):
        settings.DATASET_NAME = dataset
        settings.SEG_DIR = settings.DATA_DIR / dataset
        settings.IMG_DIR = settings.DATA_DIR / dataset.replace('seg', 'img')
        settings.RESULTS_DIR = settings.RESULTS_BASE_DIR / dataset
        settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (settings.RESULTS_DIR / 'img').mkdir(parents=True, exist_ok=True)
