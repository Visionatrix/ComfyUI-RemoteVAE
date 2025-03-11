import torch
import json
import requests

from safetensors.torch import _tobytes

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}

def remote_decode(
    endpoint: str,
    tensor: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if not height or not width:
        raise ValueError("`height` and `width` required for packed latents.")
    parameters = {
        "do_scaling": False,
        "output_type": "pt",
        "partial_postprocess": False,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
        "height": height,
        "width": width,
    }
    response = requests.post(endpoint, params=parameters, data=_tobytes(tensor, "tensor"), headers={
        "Content-Type": "tensor/binary",
        "Accept": "tensor/binary",
    })
    if not response.ok:
        raise RuntimeError(response.json())
    parameters = response.headers
    return torch.frombuffer(bytearray(response.content), dtype=DTYPE_MAP[parameters["dtype"]]).reshape(json.loads(parameters["shape"]))


class RemoteVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "vae_type": (["Flux", "HunyuanVideo"],),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "RemoteVae"

    def decode(self, samples, vae_type):
        latents = samples["samples"]
        vae_scale_factor = 8

        if vae_type == "HunyuanVideo":
            endpoint = "https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud/"
        elif vae_type == "Flux":
            endpoint = "https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/"
        else:
            raise ValueError("Unsupported VAE_type provided.")

        result = remote_decode(
            endpoint=endpoint,
            tensor=latents,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
        )

        if vae_type == "HunyuanVideo":
            out = self.postprocess_video(result)[0].permute(0, 2, 3, 1).cpu().float()
        else:
            out = self.denormalize(result).permute(0, 2, 3, 1).cpu().float()
        return (out,)

    @staticmethod
    def denormalize(images: torch.Tensor) -> torch.Tensor:
        """Denormalize an image tensor from [-1,1] to [0,1] as done in diffusers."""
        return (images * 0.5 + 0.5).clamp(0, 1)


    def postprocess_video(self, video: torch.Tensor) -> torch.Tensor:
        """Simplified postprocess_video function."""
        outputs = []
        for i in range(video.shape[0]):
            outputs.append(self.denormalize(video[i].permute(1, 0, 2, 3)))
        return torch.stack(outputs)  # [batch, frames, channels, height, width]


NODE_CLASS_MAPPINGS = {
    "RemoteVAEDecode": RemoteVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteVAEDecode": "VAE Decode (Remote)",
}
