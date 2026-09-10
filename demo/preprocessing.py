"""
Preprocessing for both stages.

The two stages need DIFFERENT preprocessing despite sharing a 224x224 input,
and getting this wrong fails silently rather than loudly:

  Stage 1 (PyTorch ResNet50)
      gamma(1.5) -> resize 224 -> ToTensor -> ImageNet normalise -> NCHW
      Must match the training/validation transform exactly.

  Stage 2 (Keras EfficientNet-B0)
      resize 224 -> RAW 0-255 float RGB -> NHWC
      The model has Rescaling(1/255) -> Normalization -> Rescaling(1/std) baked
      into its own graph. Normalising here as well would double-normalise and
      quietly wreck the predictions.
"""

import numpy as np
from PIL import Image

from demo import config

Image.MAX_IMAGE_PIXELS = None


def load_image(source):
    """Accept a path, file-like object, PIL image or numpy array -> RGB PIL."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        return Image.fromarray(source.astype(np.uint8)).convert("RGB")
    with Image.open(source) as im:
        return im.convert("RGB")      # handles JPG, PNG, RGBA, palette, etc.


def preprocess_stage1(img):
    """
    -> torch.Tensor (1, 3, 224, 224), ImageNet-normalised.

    Built from the project's own transforms so it cannot drift from training:
    stage1.dataset.eval_transform is the exact val/test transform.
    """
    from preprocessing import GammaCorrectionTransform      # project module
    from stage1.dataset import eval_transform

    tf = eval_transform(config.STAGE1_INPUT_SIZE,
                        [GammaCorrectionTransform(gamma=config.STAGE1_GAMMA)])
    return tf(img).unsqueeze(0)


def preprocess_stage2(img):
    """
    -> np.ndarray (1, 224, 224, 3), float32, values in [0, 255].

    Deliberately NOT normalised: the Keras model does that internally.
    """
    size = config.STAGE2_INPUT_SIZE
    resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)     # 0-255
    return arr[None, ...]
