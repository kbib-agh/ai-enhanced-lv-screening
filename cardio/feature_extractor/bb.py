import logging

import cv2
import numpy as np
from django.conf import settings

from cardio.helpers.colors import bgr_color
from .common import Common

_logger = logging.getLogger('features')


class BB(Common):
    def run(self) -> dict:
        try:
            self.prepare()
            data = self.bb()
        except Exception as e:
            _logger.exception(f'BB error in file {self.fid}: {e}')
            data = {'error': f'BB error: {e}'}

        if self.annot_image is not None:
            self.show_image(self.annot_image, 'bb_annot')
        return data

    def bounding_box_for_value(self, *values):
        yx = np.argwhere(np.isin(self.im, values))
        if yx.size == 0:
            return None  # brak takich pikseli

        y_min, x_min = yx.min(axis=0)
        y_max, x_max = yx.max(axis=0)

        return x_min, y_min, x_max, y_max

    def grid(self, n: int, m: int) -> list[list[float]]:
        crop_img = self.im[self.heart_box[1]:self.heart_box[3] + 1, self.heart_box[0]:self.heart_box[2] + 1]
        crop_img = np.where(crop_img == settings.HEART, 1, 0)

        scaled_img = cv2.resize(crop_img.astype(np.float32), (n, m), interpolation=cv2.INTER_AREA)

        return scaled_img.tolist()

    def bb(self) -> dict:
        self.heart_box = self.bounding_box_for_value(settings.HEART)
        cv2.rectangle(self.annot_image, self.heart_box[:2], self.heart_box[2:], color=bgr_color('red'), thickness=1)
        self.heart_width = self.heart_box[2] - self.heart_box[0]
        self.heart_height = self.heart_box[3] - self.heart_box[1]

        lungs_box = self.bounding_box_for_value(settings.RIGHT_LUNG, settings.LEFT_LUNG)
        cv2.rectangle(self.annot_image, lungs_box[:2], lungs_box[2:], color=bgr_color('green'), thickness=1)
        self.lungs_width = lungs_box[2] - lungs_box[0]
        self.lungs_height = lungs_box[3] - lungs_box[1]

        heart_grid_5x5 = self.grid(5, 5)
        heart_grid_6x6 = self.grid(6, 6)

        return {
            'heart_width': self.heart_width,
            'heart_height': self.heart_height,
            'lungs_width': self.lungs_width,
            'lungs_height': self.lungs_height,
            'heart_lungs_ratio': self.heart_width / self.lungs_width,
            'heart_ratio': self.heart_width / self.heart_height,
            'heart_grid_5x5': heart_grid_5x5,
            'heart_grid_6x6': heart_grid_6x6,
        }
