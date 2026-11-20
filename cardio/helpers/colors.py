from matplotlib.colors import to_rgba

_color_cache = {}


def bgr_color(name):
    if name in _color_cache:
        return _color_cache[name]
    a = to_rgba(name)
    value = (round(a[2] * 255), round(a[1] * 255), round(a[0] * 255))
    _color_cache[name] = value
    return value


def to_cv_color(plt_color):
    return round(plt_color[2] * 255), round(plt_color[1] * 255), round(plt_color[0] * 255)
