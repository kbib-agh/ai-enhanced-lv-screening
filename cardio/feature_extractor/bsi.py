import logging
from typing import Union

import cv2
import numpy as np
from django.conf import settings

from cardio.helpers.colors import bgr_color
from .common import Common

_logger = logging.getLogger('features')


class BSI(Common):
    def run(self) -> dict:
        try:
            self.prepare()
            self.find_heart_circle()
            data = self.bsi()
        except Exception as e:
            _logger.exception(f'BSI error in file {self.fid}: {e}')
            data = {'error': f'BSI error: {e}'}

        if self.annot_image is not None:
            self.show_image(self.annot_image, 'bsi_annot')
        return data

    def bsi(self) -> dict:
        self.heart_area = np.sum(self.im == settings.HEART)
        self.organs_area = np.sum(self.im != settings.BACKGROUND)

        self.heart_widths = self.resample(self.get_layer_widths(settings.HEART), 50)
        self.left_lung_widths = self.resample(self.get_layer_widths(settings.LEFT_LUNG), 50)
        self.right_lung_widths = self.resample(self.get_layer_widths(settings.RIGHT_LUNG), 50)
        heart_height, heart_width = self.get_layer_shape(settings.HEART)

        data = {
            'organs_area': self.organs_area,
            'heart_area': self.heart_area,
            'heart_widths_50': self.heart_widths,
            'right_lung_widths_50': self.right_lung_widths,
            'left_lung_widths_50': self.left_lung_widths,
            'center_radius': self.radius,
            'heart_width_diameter_ratio': heart_width / (2 * self.radius),
        }
        if settings.HAS_BORDER:
            thickness = 2
        else:
            thickness = 3
        cv2.circle(self.im, (self.center[1], self.center[0]), radius=self.radius, color=5, thickness=thickness)

        self.show_image(self.im, 'bsi')

        mask = cv2.inRange(self.im, settings.HEART, settings.HEART)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # consider only 4 biggest areas
        self.contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        self.fields: dict[str:Union[str, int]] = []
        for contour in self.contours:
            # Find center
            M = cv2.moments(contour)
            area = M["m00"]
            if area != 0:
                center_x = int(M["m10"] / area)
                center_y = int(M["m01"] / area)
            else:
                continue
            self.fields.append({'area': area, 'x': center_x, 'y': center_y})

        fields = sorted(self.fields, key=lambda k: k['x'])

        points = [
            ('C', self.is_center_point),
            ('U', self.is_upper_point),
            ('L', self.is_left_point),
            ('R', self.is_right_point),
        ]
        for name, fun in points:
            data[f'{name.lower()}_area'] = 0
            data[f'norm_{name.lower()}_area'] = 0
            for field in fields:
                if fun(field):
                    _logger.debug("{name}: {area}, center at {x}x{y}".format(name=name, **field))
                    cv2.circle(self.annot_image, (field['x'], field['y']), 1, bgr_color('red'), 1)
                    cv2.putText(self.annot_image, name, (field['x'] + 3, field['y']), cv2.FONT_HERSHEY_DUPLEX,
                                0.5,
                                bgr_color('red'), 1, cv2.LINE_AA)
                    data[f'{name.lower()}_area'] = field['area']
                    data[f'norm_{name.lower()}_area'] = field['area'] / self.organs_area
                    field['name'] = name
                    fields.remove(field)
                    break

        return data

    def is_center_point(self, field):
        return (field['x'] - self.center[1]) ** 2 + (field['y'] - self.center[0]) ** 2 <= self.radius ** 2

    def is_upper_point(self, field):
        return field['y'] < self.center[0]

    def is_right_point(self, field):
        return field['x'] < self.center[1]

    def is_left_point(self, field):
        return field['x'] > self.center[1]
