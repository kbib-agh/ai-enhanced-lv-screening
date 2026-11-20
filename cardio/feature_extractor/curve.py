import logging
import traceback

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress

from django.conf import settings
from .common import Common
from cardio.helpers.colors import bgr_color

_logger = logging.getLogger('features')


class Curve(Common):
    def run(self) -> dict:
        try:
            self.prepare()
            data = self.curve()
        except Exception as e:
            _logger.exception(f'Curve error in file {self.fid}: {e}')
            data = {'error': f'Curve error: {e}'}

        if self.annot_image is not None:
            self.show_image(self.annot_image, 'curve_annot')
        return data

    def find_edge_1px(self, side='left', cut_backtracking=True):
        coords = []
        h, w = self.im.shape
        # Szukamu punktów serca najbardziej wysuniętych w stronę lewego płuca
        for y in range(h):
            # idziemy wierszami i szukamy pikseli serca
            x_candidates = np.where(self.im[y] == settings.HEART)[0]
            if len(x_candidates) > 0:
                if side == 'left':
                    # wybieramy skrajny lewy punkt
                    x_max = x_candidates[-1]
                    # jeśli jest obszar niepewności (BORDER) to przesuwamy w prawo w kierunku lewego płuca
                    if settings.HAS_BORDER:
                        while x_max < w - 1 and self.im[y, x_max + 1] == settings.BORDER:
                            x_max += 1
                    if x_max == w - 1 or self.im[y, x_max + 1] == settings.LEFT_LUNG:
                        coords.append((y, x_candidates[-1]))
                else:
                    x_min = x_candidates[0]
                    if settings.HAS_BORDER:
                        while x_min > 0 and self.im[y, x_min - 1] == settings.BORDER:
                            x_min -= 1
                    if x_min == 0 or self.im[y, x_min - 1] == settings.RIGHT_LUNG:
                        coords.append((y, x_candidates[0]))
        coords = np.array(coords)

        if cut_backtracking:
            rev_coords = coords[::-1]
            # Usuwamy punkty z krzywej jeśli zawija się w kierunku serca
            cut = 0
            for dy, dx in np.diff(rev_coords, axis=0):
                if (side == 'left' and dx < 0) or (side == 'right' and dx > 0):
                    break
                cut += 1
            if cut > 0:
                coords = coords[:-cut]
        return coords

    def fit_circle(self, points):
        # Funkcja celu: odległość każdego punktu od środka - promień
        def residuals(params, x, y):
            cx, cy, r = params
            return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r

        # Punkty x i y oddzielnie
        x = points[:, 0]
        y = points[:, 1]

        # Początkowe przybliżenie: środek środka masy, promień średni
        # x_m, y_m = np.mean(x), np.mean(y)
        # r0 = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))

        mask = (self.im == settings.HEART).astype(np.uint8)
        M = cv2.moments(mask)

        if M["m00"] == 0:
            raise Exception('Missing heart')

        x_m = round(M["m10"] / M["m00"])
        y_m = round(M["m01"] / M["m00"])
        r0 = round(np.sqrt(M["m00"] / np.pi))
        cv2.circle(self.annot_image, (x_m, y_m), 1, bgr_color('plum'), 1)
        cv2.circle(self.annot_image, (x_m, y_m), r0, bgr_color('plum'), 1)
        # _logger.debug(f"Środek masy: ({y_m:.2f}, {x_m:.2f})")
        initial_guess = [x_m, y_m, r0]

        # Optymalizacja
        result = least_squares(residuals, initial_guess, args=(x, y))
        cy, cx, r = result.x
        #
        # _logger.debug(f"Dopasowany środek: ({cx:.3f}, {cy:.3f})")
        # _logger.debug(f"Dopasowany promień: {r:.3f}")
        cv2.circle(self.annot_image, (round(cx), round(cy)), round(r), bgr_color('red'), 1)
        return (cx, cy), r

    def split_points(self, points, parts_count):
        """
        Dzieli punktu należące do krzywej na `parts_count` odcinków.
        """
        n = len(points)
        part_size = round(n / parts_count)
        parts = []

        for i in range(parts_count):
            start = i * part_size
            # Upewnij się, że ostatni fragment obejmuje końcówkę
            end = (i + 1) * part_size if i < parts_count - 1 else n
            fragment = points[start:end]
            parts.append(fragment)

        return parts

    def calc_angles(self, points, parts_count):
        parts = self.split_points(points, parts_count)
        angles = []
        for i, part in enumerate(parts):
            x = part[:, 1]
            y = part[:, 0]

            if len(set(x)) == 1 or len(part) < 2:
                angle_deg = 90.0
            else:
                slope, _, _, _, _ = linregress(x, y)
                angle_rad = np.arctan(slope)
                angle_deg = np.rad2deg(angle_rad)

            angles.append(float(angle_deg))

        _logger.debug(f'Angles: {angles}')
        return angles

    def curve(self) -> dict:
        left_edge = self.find_edge_1px(side='left')
        self.draw_polyline(left_edge, bgr_color('lime'))

        # Tylko dolna połowa ścieżki
        half_edge = left_edge[len(left_edge) // 2:]
        self.draw_polyline(half_edge, bgr_color('magenta'))

        # Dopasowanie okręgu
        (cx, cy), r = self.fit_circle(half_edge)

        right_edge = self.find_edge_1px(side='right', cut_backtracking=False)
        self.draw_polyline(right_edge, bgr_color('pink'))
        right_slope, _ = np.polyfit([p[0] for p in right_edge], [p[1] for p in right_edge], 1)

        # Kąty
        angles = self.calc_angles(half_edge, settings.CURVE_ANGLES_COUNT)
        data = {
            'curve_radius': r,
            'curve_angles': angles,
            'right_slope': right_slope,
        }
        return data
