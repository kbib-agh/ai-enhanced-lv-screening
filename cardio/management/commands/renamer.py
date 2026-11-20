import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Rename and copy NII.GZ files based on pattern matching'

    def add_arguments(self, parser):
        parser.add_argument('input_dir', type=str, help='Input directory path')
        parser.add_argument('output_dir', type=str, help='Output directory path')

    def handle(self, *args, **options):
        input_path = Path(options['input_dir'])
        output_path = Path(options['output_dir'])
        output_path.mkdir(parents=True, exist_ok=True)

        for p in sorted(input_path.glob('**/*.nii.gz')):
            fname = p.relative_to(input_path)
            parent = fname.parent.name
            rx = settings.NR_REGEXP[parent]
            rxm = rx.match(fname.name)
            group = settings.DIR_NAMES.get(parent, parent)

            nr = int(rxm.group(1))
            new_name = f'{group}_{nr:04d}.nii.gz'
            new_path = output_path / new_name
            self.stdout.write(f'{p} -> {new_path}')
            shutil.copy2(p, new_path)
