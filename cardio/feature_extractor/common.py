import logging
from collections import Counter
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label, distance_transform_edt
from scipy.signal import find_peaks

from django.conf import settings
from cardio.helpers.bfs import find_path_bfs
from cardio.helpers.colors import bgr_color

_logger = logging.getLogger('features')


class Common:
    im: np.ndarray
    annot_image: np.ndarray = None
    lungs: np.ndarray
    labeled_lungs: np.ndarray
    left_lung: np.ndarray
    left_edt: np.ndarray
    right_lung: np.ndarray
    right_edt: np.ndarray
    bg: np.ndarray
    bg_edt: np.ndarray
    center: tuple[int, int]
    radius: int
    heart_area: int
    path: list[tuple[int, int]]
    heart_points: list[tuple[int, int]]
    heart_widths: list[int]
    left_lung_widths: list[int]
    right_lung_widths: list[int]

    def __init__(self, p: Path, fid: str, ):
        self.filepath = p
        self.fid = fid
        self.open_image()

    def _make_filepath(self, prefix, suffix):
        name = self.filepath.name.replace('.nii.gz', '')
        if prefix is not None:
            name = f"{prefix}_{name}"
        if suffix is not None:
            name = f"{name}_{suffix}"
        name = f'{name}.png'
        return settings.RESULTS_DIR / 'img' / name

    def show_image(self, image, prefix=None, suffix=None, level=1):
        if image.ndim == 2:
            cmap = 'gray'
        else:
            cmap = None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if level <= settings.DEBUG_LEVEL:
            filepath = self._make_filepath(prefix, suffix)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(filepath, image, cmap=cmap, format="png")

    def show_plot(self, data, prefix=None, suffix=None, level=1):
        if level <= settings.DEBUG_LEVEL:
            filepath = self._make_filepath(prefix, suffix)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.plot(data)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()

    def draw_polyline(self, path_points, color):
        cv2.polylines(self.annot_image, [np.array(path_points, np.int32)[:, [1, 0]].reshape((-1, 1, 2))],
                      isClosed=False, color=color, thickness=1)

    def open_image(self):
        self.im = nib.load(self.filepath).get_fdata()
        self.im = np.ascontiguousarray(np.round(np.squeeze(self.im)).astype(np.uint8))
        # make image for debugging annotations
        self.annot_image = cv2.cvtColor(self.im * 50, cv2.COLOR_GRAY2BGR)

    def remove_artefacts(self, layer, cluster_limit):
        artefacts = (self.im == layer).astype(np.uint8)
        labeled_artefacts, count = label(artefacts)
        # if more than one label, means artefacts with number 2 or more exists
        if count > cluster_limit:
            self.show_image(labeled_artefacts, suffix='labeled_artefacts', level=2)
            neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if i != 0 and j != 0]

            sizes = [(np.sum(labeled_artefacts == i), i) for i in range(1, count + 1)]
            # skip biggest area
            for size, lbl in sorted(sizes, reverse=True)[cluster_limit:]:
                # find clusters of artefact and fill it with neighbor color
                for r, c in zip(*np.where(labeled_artefacts == lbl)):
                    values = []
                    for i, j in neighbors:
                        new_r, new_c = r + i, c + j
                        if (0 <= new_r < self.im.shape[0] and 0 <= new_c < self.im.shape[1] and
                                self.im[new_r, new_c] != layer):
                            values.append(self.im[new_r, new_c])
                    # wartość występująca najczęściej
                    self.im[r, c] = Counter(values).most_common(1)[0][0]

    def get_layer_widths(self, layer: int) -> list[int]:
        mask = (self.im == layer)
        diff = np.diff(mask.astype(np.int8), axis=1)

        widths = []
        for row in diff:
            starts = np.where(row == 1)[0]
            if not starts.size:
                continue

            start = starts[0]
            end = np.where(row == -1)[0][-1]
            widths.append(end - start)
        return widths

    def get_layer_shape(self, layer: int) -> tuple[int, int]:
        # Znajdź pozycje wszystkich True
        rows, cols = np.where(self.im == layer)

        # Oblicz wymiary
        if len(rows) > 0:
            height = int(rows.max() - rows.min() + 1)
            width = int(cols.max() - cols.min() + 1)
            return height, width
        return 0, 0

    def remove_border(self):
        # rozszerzamy dopóki ramka nie zniknie
        while self.im.max() == settings.ORIG_BORDER:
            # Rozszerzamy płuca i potem serce
            for target_value in [settings.ORIG_LUNGS, settings.ORIG_HEART]:
                # Tworzymy maskę dla wartości target
                mask = (self.im == target_value).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)

                # Wstawiamy wartość `target_value` tam, gdzie pojawiły się nowe piksele po dylatacji
                self.im[(dilated_mask == 1) & (self.im == settings.ORIG_BORDER)] = target_value

    def swap_heart_lungs(self):
        mapa = np.array([settings.BACKGROUND, settings.ORIG_HEART, settings.ORIG_LUNGS, settings.BORDER])
        self.im = mapa[self.im].astype(np.uint8)

    def prepare(self):
        self.im = np.pad(self.im, pad_width=1, mode='constant', constant_values=settings.BACKGROUND)
        if settings.HAS_BORDER:
            self.swap_heart_lungs()

        # remove small black artifacts from image
        self.remove_artefacts(settings.BACKGROUND, 1)
        # remove small lung dots artifacts from image
        self.remove_artefacts(settings.ORIG_LUNGS, 2)
        # remove small heart dots artifacts from image
        self.remove_artefacts(settings.ORIG_HEART, 1)

        # Changing labels from: lungs => 1, heart => 2
        # to: lungs => 1 and 2, heart => 3
        # remove heart
        self.lungs = np.copy(self.im)
        self.lungs[np.isin(self.lungs, [settings.ORIG_HEART, settings.BORDER])] = settings.BACKGROUND
        # label right and left lung, transposition needed to label from right to left
        self.labeled_lungs, _ = label(self.lungs.T)
        self.labeled_lungs = self.labeled_lungs.T

        # change heart to 3
        self.im[self.im == settings.ORIG_HEART] = settings.HEART
        # coby labeled lungs to image
        self.im[self.labeled_lungs != settings.BACKGROUND] = np.copy(
            self.labeled_lungs[self.labeled_lungs != settings.BACKGROUND])

        # update after renumerating
        self.annot_image = cv2.cvtColor(self.im * 50, cv2.COLOR_GRAY2BGR)

    def find_heart_circle(self):
        # calculate distances from visually left lung
        self.left_lung = (self.im != settings.LEFT_LUNG).astype(np.int8)
        self.show_image(self.left_lung, suffix='left_lung', level=2)
        self.left_edt = distance_transform_edt(self.left_lung)
        self.show_image(self.left_edt, suffix='left_edt', level=2)

        # calculate distances from visually right lung
        self.right_lung = (self.im != settings.RIGHT_LUNG).astype(np.int8)
        self.show_image(self.right_lung, suffix='right_lung', level=2)

        self.right_edt = distance_transform_edt(self.right_lung)
        self.show_image(self.right_edt, suffix='right_edt', level=2)

        # calculate distances from background
        self.bg = (self.im != settings.BACKGROUND).astype(np.int8)
        self.show_image(self.bg, suffix='bg', level=2)
        self.bg_edt = distance_transform_edt(self.bg)
        self.show_image(self.bg_edt, suffix='bg_edt', level=2)

        # find a path separating right and left lung
        self.path = self.find_path(self.left_edt, self.right_edt, bgr_color('gold'), prefix='diff_left_right')

        self.center, self.radius, self.heart_points = self.find_circle(self.path, self.left_edt, self.bg_edt,
                                                                       bgr_color('aqua'),
                                                                       suffix='diff_left_bg_plot')

    def find_path(self, left_edt, right_edt, draw_color, start=None, end=None, prefix=None):

        # find a path separating right and left lung
        diff = np.abs(left_edt - right_edt)
        self.show_image(diff, suffix=prefix, level=2)
        if start is None:
            start = (0, np.argmin(diff[0]))
        if end is None:
            end = (diff.shape[0] - 1, np.argmin(diff[-1]))
        path = find_path_bfs(diff, start, end)

        self.draw_polyline(path, draw_color)

        return path

    def find_circle(self, path, edt1, edt2, draw_color, suffix):
        # iterate along the path and collect differnces between distance from lung and distance from background
        differences = []
        points = []
        for p in path:
            if self.im[p[0], p[1]] == settings.HEART:
                differences.append(abs(edt1[p] - edt2[p]))
                points.append(p)

        # minimum difference indicates the center of circle tangent to lungs and lower border of the heart
        differences = np.array(differences)
        minimum, _ = find_peaks(-differences, width=10)
        self.show_plot(differences, suffix=suffix, level=2)

        for min_idx in minimum:
            y, x = points[min_idx]
            cv2.circle(self.annot_image, (x, y), radius=2, color=bgr_color('lime'), thickness=-1)

        # if minimum.size != 2:
        #     raise Exception(f'Found centers {minimum.size} != 2')

        center_idx = minimum[-1]
        center = points[center_idx]
        radius = round(np.mean([edt1[center], edt2[center], edt2[center]]))
        _logger.debug(f'tangent circle at {center}, self.radius = {radius}')

        cv2.circle(self.annot_image, (center[1], center[0]), radius=radius, color=draw_color, thickness=1)

        return center, radius, points

    def resample(self, arr: np.ndarray | list, new_length: int = 10):
        old_indices = np.arange(0, len(arr))
        new_indices = np.linspace(0, len(arr) - 1, new_length)
        return np.interp(new_indices, old_indices, arr)
