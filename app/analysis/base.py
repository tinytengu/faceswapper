from abc import ABC, abstractmethod

import numpy as np
from insightface.app.common import Face


class FaceAnalyzer(ABC):
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> list[Face]:
        raise NotImplementedError
