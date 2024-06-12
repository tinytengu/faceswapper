from typing import Any

import cv2
import numpy as np
from PIL import Image


def pil_to_numpy(image: Image.Image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def numpy_to_pil(image: np.ndarray | Any):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
