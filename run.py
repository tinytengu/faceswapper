import sys
import pathlib

sys.path.append("./CodeFormer/CodeFormer")

import cv2
import numpy as np

from app.analysis.insightface import InsightFaceAnalyzer
from app.swapping.insightface import InsightFaceSwapper

INSIGHTFACE_ROOT = pathlib.Path.home() / ".insightface"
INSIGHTFACE_MODELS_DIR = INSIGHTFACE_ROOT / "models"


def main():
    source_img = cv2.imread("data/lisa.jpeg")
    target_img = cv2.imread("data/jisoo.jpeg")

    analyzer = InsightFaceAnalyzer(model_name="buffalo_l", root_dir=INSIGHTFACE_ROOT)
    swapper = InsightFaceSwapper(
        model_name="inswapper_128.onnx", root_dir=INSIGHTFACE_ROOT
    )

    source_img_face = analyzer.detect_faces(source_img)[0]
    target_img_face = analyzer.detect_faces(target_img)[0]

    swapped_image = swapper.swap_face(
        image=target_img,
        source_face=source_img_face,
        target_face=target_img_face,
    )

    if isinstance(swapped_image, np.ndarray):
        cv2.imwrite("result.png", swapped_image)
    else:
        raise ValueError("Swapped image is not a numpy array")

    # face_analyzer = get_face_analyzer(
    #     model_name="buffalo_l",
    #     providers=["CPUExecutionProvider"],
    #     # providers=onnxruntime.get_available_providers(),
    # )

    # face_swapper = get_face_swap_model(model_path=MODELS_DIR / "inswapper_128.onnx")


if __name__ == "__main__":
    main()
