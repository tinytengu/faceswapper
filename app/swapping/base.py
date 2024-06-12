from abc import ABC, abstractmethod

import numpy as np
from insightface.app.common import Face


class FaceSwapper(ABC):
    @abstractmethod
    def swap_face(self, image: np.ndarray, source_face: Face, target_face: Face):
        raise NotImplementedError
