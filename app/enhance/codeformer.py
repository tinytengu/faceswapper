import pathlib

import numpy as np
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper


from .realesrgan_upsampler import RealESRGANUpsampler


class CodeFormerEnhancer:
    def __init__(
        self, root: pathlib.Path, detection_model: str = "retinaface_resnet50"
    ):
        self.root = root
        self.detection_model = detection_model

        self.verify_checkpoints()

    def verify_checkpoints(self):
        pretrain_model_url = {
            "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
            "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        }

        print("Checking", self.root / "weights/CodeFormer/codeformer.pth")

        if not (self.root / "weights/CodeFormer/codeformer.pth").exists():
            load_file_from_url(
                url=pretrain_model_url["codeformer"],
                model_dir=str(self.root / "weights/CodeFormer"),
                progress=True,
                file_name=None,
            )

        if not (self.root / "weights/facelib/detection_Resnet50_Final.pth").exists():
            load_file_from_url(
                url=pretrain_model_url["detection"],
                model_dir=str(self.root / "weights/facelib"),
                progress=True,
                file_name=None,
            )

        if not (self.root / "weights/facelib/parsing_parsenet.pth").exists():
            load_file_from_url(
                url=pretrain_model_url["parsing"],
                model_dir=str(self.root / "weights/facelib"),
                progress=True,
                file_name=None,
            )

        if not (self.root / "weights/realesrgan/RealESRGAN_x2plus.pth").exists():
            load_file_from_url(
                url=pretrain_model_url["realesrgan"],
                model_dir=str(self.root / "weights/realesrgan"),
                progress=True,
                file_name=None,
            )

    def get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def get_codeformer_net(self, device: torch.device | None = None):
        device = device or self.get_device()

        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        ckpt_path = self.root / "weights/CodeFormer/codeformer.pth"

        checkpoint = torch.load(str(ckpt_path))["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()

        return net

    def normalize_upscale_factor(self, upscale_factor: int, img: np.ndarray):
        # avoid memory exceeded due to too large upscale_factor
        upscale_factor = max(min(upscale_factor, 4), 1)

        # avoid memory exceeded due to too large img resolution
        if upscale_factor > 2 and max(img.shape[:2]) > 1000:
            upscale_factor = 2

        # avoid memory exceeded due to too large img resolution
        if max(img.shape[:2]) > 1500:
            upscale_factor = 1

        return upscale_factor

    def enhance_faces(
        self,
        img: np.ndarray,
        upscale_factor: int = 1,
        fidelity: float = 0.5,
        upsampler: RealESRGANUpsampler | None = None,
        enhance_background: bool = False,
        device: torch.device | None = None,
        net: torch.nn.Module | None = None,
    ):
        device = device or self.get_device()
        codeformer_net = net or self.get_codeformer_net(device=device)

        upscale_factor = self.normalize_upscale_factor(upscale_factor, img)

        face_helper = FaceRestoreHelper(
            upscale_factor=upscale_factor,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=self.detection_model,
            save_ext="png",
            use_parse=True,
        )

        face_helper.read_image(img)

        # Get face landmarks for each face
        face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )

        # Align and warp each face
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for cropped_face in face_helper.cropped_faces:
            # prepare data
            face_tensor = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            if not isinstance(face_tensor, torch.Tensor):
                raise ValueError("Face tensor is None")

            normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            face_tensor = face_tensor.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(face_tensor, w=fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))  # type: ignore
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(face_tensor, rgb2bgr=True, min_max=(-1, 1))  # type: ignore

            restored_face: np.ndarray = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        face_helper.get_inverse_affine(None)

        if enhance_background:
            if not upsampler:
                raise ValueError("Upsampler is None")

            bg_img = self.enhance_background(
                img=img,
                upsampler=upsampler,
                upscale_factor=upscale_factor,
            )
        else:
            bg_img = img

        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=False,
            face_upsampler=upsampler,
        )

        return restored_img

    def enhance_background(
        self,
        img: np.ndarray,
        upsampler: RealESRGANUpsampler,
        upscale_factor: int = 1,
    ):
        upscale_factor = self.normalize_upscale_factor(upscale_factor, img)
        return upsampler.enhance(img=img, outscale=upscale_factor)[0]
