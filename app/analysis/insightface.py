import pathlib

import numpy as np
import insightface
from insightface.app.common import Face
import onnxruntime


from .base import FaceAnalyzer


class InsightFaceAnalyzer(FaceAnalyzer):
    def __init__(
        self,
        model_name: str,
        det_size: tuple[int, int] = (320, 320),
        root_dir: pathlib.Path | None = None,
        providers: list[str] | None = None,
    ):
        root_dir = root_dir or pathlib.Path("~/.insightface").expanduser()
        providers = providers or onnxruntime.get_available_providers()

        self.analyzer = insightface.app.FaceAnalysis(
            name=model_name, root=str(root_dir), providers=providers
        )
        self.analyzer.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, image: np.ndarray) -> list[Face]:
        return self.analyzer.get(image)
