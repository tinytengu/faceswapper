import sys
import pathlib

sys.path.append("./CodeFormer")

import cv2
import numpy as np

from app.analysis.insightface import InsightFaceAnalyzer
from app.swapping.insightface import InsightFaceSwapper
from app.enhance.codeformer import CodeFormerEnhancer

INSIGHTFACE_ROOT = pathlib.Path.home() / ".insightface"
INSIGHTFACE_MODELS_DIR = INSIGHTFACE_ROOT / "models"
CODEFORMER_ROOT = pathlib.Path("./CodeFormer/").resolve()


def main():
    source_img = cv2.imread("data/lisa.jpeg")
    target_img = cv2.imread("data/jisoo.jpeg")

    analyzer = InsightFaceAnalyzer(model_name="buffalo_l", root_dir=INSIGHTFACE_ROOT)
    swapper = InsightFaceSwapper(
        model_name="inswapper_128.onnx", root_dir=INSIGHTFACE_ROOT
    )
    enhancer = CodeFormerEnhancer(CODEFORMER_ROOT)

    source_img_face = analyzer.detect_faces(source_img)[0]
    target_img_face = analyzer.detect_faces(target_img)[0]

    swapped_image = swapper.swap_face(
        image=target_img,
        source_face=source_img_face,
        target_face=target_img_face,
    )

    if not isinstance(swapped_image, np.ndarray):
        raise Exception("Swapping failed")

    swapped_image = enhancer.enhance_faces(
        img=swapped_image,
        upscale_factor=1,
        fidelity=0.5,
    )

    if isinstance(swapped_image, np.ndarray):
        cv2.imwrite("result.png", swapped_image)
    else:
        raise ValueError("Swapped image is not a numpy array")


if __name__ == "__main__":
    main()
