import logging

import cv2
import numpy as np
from django.conf import settings
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from cardio.feature_extractor.bb import BB
from cardio.feature_extractor.bsi import BSI
from cardio.feature_extractor.curve import Curve
from cardio.feature_extractor.lsi import LSI
from cardio.helpers.features_labels import get_label_map
from cardio.helpers.parallel import parallel
from ._base import Command as BaseCommand
from ...helpers.colors import bgr_color
from ...helpers.files import open_nii
from ...helpers.plot import create_fig
from ...models import Feature

_logger = logging.getLogger('features')

MASK_COLORS = [
    (settings.RIGHT_LUNG, 'tab:blue', 'Right lung'),
    (settings.LEFT_LUNG, 'tab:cyan', 'Left lung'),
    (settings.HEART, 'tomato', 'Heart'),
    (settings.BORDER, 'gold', 'Uncertain region'),
]


def mask_labels(p, fid):
    # fresh labels after prepare(); run() of BSI/LSI draws circles and paths into extractor.im
    extractor = BSI(p, fid)
    extractor.prepare()
    return extractor.im


def overlay_masks(img, labels, alpha=0.4):
    # img: BGR uint8, labels: label map of the same shape
    out = img.astype(np.float32)
    for value, color, _ in MASK_COLORS:
        mask = labels == value
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array(bgr_color(color))
    return np.round(out).astype(np.uint8)


def image(p, extractor, colored_masks=False):

    orig_path = settings.IMG_DIR / p.name
    if not orig_path.exists():
        return

    orig_img = open_nii(orig_path, normalize=True)
    # same 1px padding as in Common.prepare(), so extractor coordinates match the image
    orig_img = np.pad(orig_img, pad_width=1, mode='edge')
    img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    if colored_masks:
        img = overlay_masks(img, mask_labels(p, extractor.fid))

    # płuca
    mask = cv2.inRange(extractor.im, settings.RIGHT_LUNG, settings.LEFT_LUNG)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr_color('black'), 1)

    # tło
    mask = cv2.inRange(extractor.im, settings.RIGHT_LUNG, settings.BORDER)
    cv2.circle(mask, (extractor.center[1], extractor.center[0]), radius=extractor.radius, color=1, thickness=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr_color('black'), 1)
    return img


def unpad(img):
    # remove 1px padding added in image()
    return np.ascontiguousarray(img[1:-1, 1:-1])


def draw_bsi(outdir, p, dataset, fid, colored_masks=False):
    bsi = BSI(p, fid)
    bsi.run()
    img = image(p, bsi, colored_masks)
    if img is None:
        return

    cv2.drawContours(img, bsi.contours, -1, bgr_color('red'), 1)
    cv2.circle(img, (bsi.center[1], bsi.center[0]), radius=bsi.radius+1, color=bgr_color('blue'), thickness=1)
    for field in bsi.fields:
        cv2.circle(img, (field['x'], field['y']), 1, bgr_color('red'), 1)
        cv2.putText(img, field['name'], (field['x'] + (-13 if field['name'] == 'R' else 3), field['y']), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, bgr_color('red'), 1, cv2.LINE_AA)

    cv2.polylines(img, [np.array(bsi.path, np.int32)[:, [1, 0]].reshape((-1, 1, 2))],
                  isClosed=False, color=bgr_color('green'), thickness=1)

    img = cv2.cvtColor(unpad(img), cv2.COLOR_BGR2RGB)
    name = 'bsi_masks' if colored_masks else 'bsi'
    plt.imsave(outdir / f'{name}_{fid}.svg', img)
    plt.imsave(outdir / f'{name}_{fid}.png', img)


def draw_lsi(outdir, p, dataset, fid, colored_masks=False):
    lsi = LSI(p, fid)
    lsi.run()
    img = image(p, lsi, colored_masks)
    if img is None:
        return

    # cv2.drawContours(img, bsi.contours, -1, bgr_color('blue'), 1)
    cv2.circle(img, (lsi.center[1], lsi.center[0]), radius=lsi.radius, color=bgr_color('blue'), thickness=1)
    cv2.circle(img, (lsi.left_center[1], lsi.left_center[0]), radius=lsi.left_radius, color=bgr_color('red'), thickness=1)

    cv2.polylines(img, [np.array(lsi.path, np.int32)[:, [1, 0]].reshape((-1, 1, 2))],
                  isClosed=False, color=bgr_color('green'), thickness=1)
    cv2.polylines(img, [np.array(lsi.left_path, np.int32)[:, [1, 0]].reshape((-1, 1, 2))],
                  isClosed=False, color=bgr_color('purple'), thickness=1)
    cv2.polylines(img, [np.array(lsi.circle_apex_path, np.int32)[:, [1, 0]].reshape((-1, 1, 2))],
                  isClosed=False, color=bgr_color('magenta'), thickness=1)

    img = cv2.cvtColor(unpad(img), cv2.COLOR_BGR2RGB)
    name = 'lsi_masks' if colored_masks else 'lsi'
    plt.imsave(outdir / f'{name}_{fid}.svg', img)
    plt.imsave(outdir / f'{name}_{fid}.png', img)


def draw_masks(outdir, p, dataset, fid, alpha=0.4):
    orig_path = settings.IMG_DIR / p.name
    if not orig_path.exists():
        return

    # remove 1px padding added in prepare() to align masks with the original image
    labels = unpad(mask_labels(p, fid))

    orig_img = open_nii(orig_path, normalize=True)
    img = np.stack([orig_img] * 3, axis=-1).astype(np.float32) / 255

    handles = []
    for value, color, name in MASK_COLORS:
        mask = labels == value
        if not mask.any():
            continue
        img[mask] = (1 - alpha) * img[mask] + alpha * np.array(to_rgb(color))
        handles.append(Patch(facecolor=color, alpha=alpha, edgecolor=color, label=name))

    h, w = labels.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100 + 0.6), dpi=100)
    ax.imshow(np.clip(img, 0, 1))
    # region borders make adjacent classes easier to tell apart
    for value, color, _ in MASK_COLORS:
        contours, _ = cv2.findContours((labels == value).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            xy = np.vstack([contour[:, 0, :], contour[:1, 0, :]])
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.8)
    ax.set_axis_off()
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(handles),
              frameon=False, fontsize=9)
    fig.savefig(outdir / f'masks_{fid}.svg', bbox_inches='tight')
    fig.savefig(outdir / f'masks_{fid}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


class Command(BaseCommand):
    help = 'Draw images for publication'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        settings.DATASETS=['seg03.1']
        for dataset in settings.DATASETS:
            _logger.info(f'Dataset: {dataset}')
            self.choose_dataset(dataset)
            settings.HAS_BORDER = dataset.endswith('.1')

            outdir = settings.RESULTS_DIR / 'pub_images'
            outdir.mkdir(parents=True, exist_ok=True)

            if not dataset.endswith('.1'):
                continue

            fnames = sorted(settings.SEG_DIR.glob('**/newest_0036.nii.gz'))

            for p in fnames:
                fname = p.relative_to(settings.SEG_DIR)
                fid = fname.name.replace('.nii.gz', '')
                if fid in settings.EXCLUDE:
                    continue
                print(fid)

                for colored_masks in (False, True):
                    draw_lsi(outdir, p, dataset, fid, colored_masks)
                    draw_bsi(outdir, p, dataset, fid, colored_masks)
                draw_masks(outdir, p, dataset, fid)
