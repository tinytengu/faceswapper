import pathlib

import numpy as np
import insightface
from insightface.model_zoo.inswapper import INSwapper

from app.utils import get_insightface_root

from .base import FaceSwapper, Face


class InsightFaceSwapper(FaceSwapper):
    def __init__(self, model_name: str, root_dir: pathlib.Path | None = None):
        root_dir = root_dir or get_insightface_root()

        self.model = insightface.model_zoo.get_model(
            str(root_dir / "models" / model_name)
        )

    def swap_face(self, image: np.ndarray, source_face: Face, target_face: Face):
        if not isinstance(self.model, INSwapper):
            raise ValueError("INSwapper model is required")

        return self.model.get(
            img=image, target_face=target_face, source_face=source_face, paste_back=True
        )
