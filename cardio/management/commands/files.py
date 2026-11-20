from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from django.conf import settings
from django.core.management.base import BaseCommand

from cardio.helpers.features_labels import get_features
from cardio.helpers.files import open_nii
from cardio.helpers.parallel import parallel
from cardio.helpers.plot import create_fig


class Command(BaseCommand):
    help = 'Manage files: export, compare, merge results, or export labels'

    def add_arguments(self, parser):
        parser.add_argument('action', type=str,
                            choices=['export', 'merge_results', 'compare', 'export_labels'],
                            help='Action to perform')

    def handle(self, *args, **options):
        action = options['action']

        for dataset in settings.DATASETS:
            self.choose_dataset(dataset)

            if action == 'export':
                self.export()
                return

            if action == 'merge_results':
                self.merge_results()
                return

            data = get_features()

            if action == 'compare':
                compare_dir = settings.RESULTS_DIR / 'compare'
                compare_dir.mkdir(parents=True, exist_ok=True)
                parallel(compare, [(compare_dir, index, row) for index, row in data.iterrows()])
                return

            if action == 'export_labels':
                self.export_labels(data)

    def export(self):
        for p in settings.IMG_DIR.glob('*.nii.gz'):
            self.stdout.write(str(p))
            img = open_nii(p, normalize=True)
            plt.imsave(p.with_name(p.name.replace('nii.gz', 'png')), img, cmap='gray')
        for p in settings.SEG_DIR.glob('*.nii.gz'):
            self.stdout.write(str(p))
            img = open_nii(p)
            plt.imsave(p.with_name(p.name.replace('nii.gz', 'png')), img, cmap='gray')

    def merge_results(self):
        for filename in ['final_results', 'pearson', 'roc_results']:
            dfs = []
            for p in settings.RESULTS_BASE_DIR.glob(f'**/{filename}.xlsx'):
                self.stdout.write(str(p))
                df = pd.read_excel(p)
                if filename == 'pearson':
                    df.set_index('Feature', inplace=True)
                    df = df.T
                rel_p = p.relative_to(settings.RESULTS_BASE_DIR)
                for nr, part in enumerate(rel_p.parent.parts):
                    df.insert(nr, nr, part)
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            df_all.to_excel(settings.RESULTS_BASE_DIR / f"merged_{filename}.xlsx", index=False)

    def export_labels(self, data):
        base_dir = settings.IMG_DIR.with_name(settings.IMG_DIR.name.replace('img', 'lbl'))

        for label in settings.LABELS_COLUMN:
            label_dir = base_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            y = data[label]
            X = data[['Filename', label]]

            for seed in settings.SEEDS:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed, stratify=y
                )

                train_df = pd.DataFrame(X_train).set_index('Filename')
                train_df.sort_values(by='Filename', inplace=True)
                train_df.to_csv((label_dir / f'{seed}_train').with_suffix('.csv'))
                test_df = pd.DataFrame(X_test).set_index('Filename')
                test_df.sort_values(by='Filename', inplace=True)
                test_df.to_csv((label_dir / f'{seed}_test').with_suffix('.csv'))


def compare(compare_dir: Path, index: str, row: pd.Series):
    print(index)

    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = create_fig(2, 3, width=3000, height=2000)

    fs = 20
    orig_path = settings.IMG_DIR / f'{index}.nii.gz'
    if orig_path.exists():
        orig_img = open_nii(orig_path, normalize=True)
        ax0.imshow(orig_img, cmap='gray')
        ax0.set_title('Filename: {Filename}, LVEDV: {label_lvedv}, LVM: {label_lvm}, OR: {label_or}'.format(**row),
                      fontsize=fs)
        ax0.axis('off')

    seg_path = settings.SEG_DIR / f'{index}.nii.gz'
    if seg_path.exists():
        seg_img = open_nii(seg_path)
        ax1.imshow(seg_img, cmap='gray')
        ax1.set_title('Organs area: {organs_area}, Heart area: {heart_area}'.format(**row), fontsize=fs)
        ax1.axis('off')

    bb_path = settings.RESULTS_DIR / f'img/bb_annot_{index}.png'
    if bb_path.exists():
        bb_annot_img = Image.open(bb_path)
        ax2.imshow(bb_annot_img)
        ax2.set_title('Heart/Lungs ratio: {heart_lungs_ratio}'.format(**row), fontsize=fs)
        ax2.axis("off")

    lsi_path = settings.RESULTS_DIR / f'img/lsi_annot_{index}.png'
    if lsi_path.exists():
        lsi_annot_img = Image.open(lsi_path)
        ax3.imshow(lsi_annot_img)
        ax3.set_title('Slope: {slope}, Angle: {angle}, Left radius: {left_radius}'.format(**row), fontsize=fs)
        ax3.axis("off")

    bsi_path = settings.RESULTS_DIR / f'img/bsi_annot_{index}.png'
    if bsi_path.exists():
        bsi_annot_img = Image.open(bsi_path)
        ax4.imshow(bsi_annot_img)
        ax4.set_title('C: {c_area}, U: {u_area}, L: {l_area}, R: {r_area}'.format(**row), fontsize=fs)
        ax4.axis("off")

    curve_path = settings.RESULTS_DIR / f'img/curve_annot_{index}.png'
    if curve_path.exists():
        curve_annot_img = Image.open(curve_path)
        ax5.imshow(curve_annot_img)
        ax5.set_title('Curve radius: {curve_radius} {angles}'.format(**row, angles=', '.join([
            str(row[f'curve_angle_{i}']) for i in range(settings.CURVE_ANGLES_COUNT)])), fontsize=fs)
        ax5.axis("off")

    fig.tight_layout()
    fig.savefig(compare_dir / f'{index}.png')
    plt.close(fig)
