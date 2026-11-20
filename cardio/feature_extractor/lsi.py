import logging
import math
import traceback
import warnings

import cv2
import numpy as np
from django.conf import settings
from scipy.ndimage import distance_transform_edt

from cardio.helpers.colors import bgr_color
from cardio.helpers.dijkstra import find_path_dijkstra
from .common import Common

_logger = logging.getLogger('features')


class LSI(Common):
    def run(self) -> dict:
        try:
            self.prepare()
            self.find_heart_circle()
            data = self.lsi()
        except Exception as e:
            _logger.exception(f'LSI error in file {self.fid}: {e}')
            data = {'error': f'LSI error: {e}'}

        if self.annot_image is not None:
            self.show_image(self.annot_image, 'lsi_annot')
        return data

    def lsi(self) -> dict:
        data = {}
        for p in self.heart_points:
            self.im[p] = 5

        # fill image to the separating path
        right_half = np.copy(self.im)
        for p in self.path:
            right_half[p[0], :p[1]] = settings.RIGHT_LUNG

        # calculate distances from right half image
        right_half = (right_half != settings.RIGHT_LUNG).astype(np.int8)
        right_half_edt = distance_transform_edt(right_half)
        self.show_image(right_half, suffix='right_half', level=2)
        self.show_image(right_half_edt, suffix='right_half_edt', level=2)

        # find a path separating right half image and left lung
        self.left_path = self.find_path(self.left_edt, right_half_edt, bgr_color('violet'),
                                        prefix='diff_left_right_half')

        # find circle tangent to lung, right half and lower border of the heart
        self.left_center, self.left_radius, left_points = self.find_circle(self.left_path, self.left_edt, self.bg_edt,
                                                                           bgr_color('indigo'),
                                                                           suffix='diff_left_bg')

        cv2.circle(self.im, (self.left_center[1], self.left_center[0]), self.left_radius, 6, 1)

        # find paths connecting circle centers and central line of heart
        diff_left_bg = np.abs(self.left_edt - self.bg_edt)
        circles_path = self.find_path(self.left_edt, self.bg_edt, bgr_color('fuchsia'), start=self.left_center,
                                      end=self.path, prefix='diff_left_bg')

        left_circle_apex_path = find_path_dijkstra(diff_left_bg, self.left_center,
                                                        [settings.LEFT_LUNG, settings.BACKGROUND], self.im)
        self.circle_apex_path = circles_path[::-1] + left_circle_apex_path
        self.draw_polyline(left_circle_apex_path, bgr_color('blue'))

        for p in self.circle_apex_path:
            if self.im[p] == settings.HEART:  # heart
                self.im[p] = 7

        index_of_center = self.circle_apex_path.index(self.left_center)
        index_of_junction = self.path.index(self.circle_apex_path[0])
        path_pts = self.path[max(index_of_junction - settings.MARGIN, 0):index_of_junction + settings.MARGIN]
        self.draw_polyline(path_pts, bgr_color('green'))

        slope1, _ = np.polyfit([p[0] for p in path_pts], [p[1] for p in path_pts], 1)
        path_pts = self.circle_apex_path[max(index_of_center - settings.MARGIN, 0):index_of_center + settings.MARGIN]
        self.draw_polyline(path_pts, bgr_color('green'))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=np.exceptions.RankWarning)
            slope2, _ = np.polyfit([p[0] for p in path_pts], [p[1] for p in path_pts], 1)
        tan_theta = abs((slope1 - slope2) / (1 + slope1 * slope2))
        theta_radians = math.atan(tan_theta)
        theta_degrees = math.degrees(theta_radians)

        _logger.debug(f'circle at {self.center}, lsi = {self.radius}, ia = {theta_degrees}')

        self.show_image(self.im, 'lsi')

        cxp = self.circle_apex_path[0]
        points = [p for p in self.heart_points if p[0] > cxp[0]]
        self.draw_polyline(points, bgr_color('orange'))
        # Fit line to point under cross point
        vy, vx, y0, x0 = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        slope = float(vy / vx)
        intercept = float(y0 - slope * x0)
        angle_radians = math.atan(slope)
        _logger.debug(f'Slope: {slope:.4f}, angle: {angle_radians:.4f} rad')
        # Draw line
        y2 = self.circle_apex_path[-1][0]
        x2 = int((self.circle_apex_path[-1][0] - intercept) / slope)
        cv2.line(self.annot_image, (cxp[1], cxp[0]), (x2, y2), bgr_color('brown'), 1)

        data.update({
            'slope': slope,
            'angle': theta_radians,
            'left_radius': self.left_radius,
        })

        return data
