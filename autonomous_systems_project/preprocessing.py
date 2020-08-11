import cv2


def rescale(shape: tuple):
    def rescale_inner(observation):
        nonlocal shape
        return cv2.resize(observation, shape, interpolation=cv2.INTER_AREA)

    return rescale_inner


def crop(y_start: int, y_end: int, x_start: int, x_end):
    def crop_inner(observation):
        nonlocal y_start, y_end, x_start, x_end
        return observation[y_start:y_end, x_start:x_end]

    return crop_inner
