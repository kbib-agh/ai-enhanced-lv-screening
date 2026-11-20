import logging

from django.conf import settings

from cardio.feature_extractor.bb import BB
from cardio.feature_extractor.bsi import BSI
from cardio.feature_extractor.curve import Curve
from cardio.feature_extractor.lsi import LSI
from cardio.helpers.features_labels import get_label_map
from cardio.helpers.parallel import parallel
from ._base import Command as BaseCommand
from ...models import Feature

_logger = logging.getLogger('features')


def worker(p, dataset, fid, data_dict) -> bool:
    bsi = BSI(p, fid).run()
    lsi = LSI(p, fid).run()
    bb = BB(p, fid).run()
    curve = Curve(p, fid).run()
    result = bsi | lsi | bb | curve

    if 'error' in result:
        return False
    else:
        Feature.objects.update_or_create(dataset=dataset, fid=fid, defaults=data_dict | result)
        return True


class Command(BaseCommand):
    help = 'Extract features from segmented NII files'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):

        for dataset in settings.DATASETS:
            _logger.info(f'Dataset: {dataset}')
            self.choose_dataset(dataset)

            settings.HAS_BORDER = dataset.endswith('.1')

            fnames = sorted(settings.SEG_DIR.glob('**/*.nii.gz'))
            y_map = get_label_map()

            data = {}
            parameters = []

            for p in fnames:
                fname = p.relative_to(settings.SEG_DIR)
                fid = fname.name.replace('.nii.gz', '')
                if fid in settings.EXCLUDE:
                    continue

                group, nr = fid.split('_')
                nr = int(nr)
                label = y_map[fid]
                data_dict = {
                                'group': group,
                                'nr': nr,
                                'filename': fname,
                            } | label
                parameters.append((p, dataset, fid, data_dict))

            worker_results = parallel(worker, parameters)

            errors = 0
            for result in worker_results:
                if not result:
                    errors += 1

            self.stdout.write(self.style.WARNING(f'\n{errors} ERRORS'))
            if errors == 0:
                self.stdout.write(self.style.SUCCESS('Feature extraction completed successfully'))

            self.stdout.write(self.style.SUCCESS(f'Features in DB: {Feature.objects.count()}'))
