"""
Preprocessing / contrast-enhancement transforms.

Each class is a callable usable inside torchvision.transforms.Compose([...]).
They expect a PIL.Image (as produced by torchvision's default ImageFolder
loader) and return a PIL.Image, so they can be dropped in front of
transforms.Resize() in any pipeline.

These are used by BOTH the train_*.py and evaluate_*.py scripts via
common.py, so a given method (e.g. CLAHE) is guaranteed to see the exact
same preprocessing at train time and at test time.

Note on method design (not changed here, just flagged for your write-up):
CLAHETransform and HistogramEqualizationTransform only touch the
luminance channel (LAB "L" / YCrCb "Y"), leaving color untouched.
GammaCorrectionTransform is applied independently to all three RGB
channels. That's a standard way to implement each method, but it does
mean they aren't operating in identical color spaces -- worth a sentence
in your methodology section.
"""

import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    """Contrast Limited Adaptive Histogram Equalization on the L channel (LAB)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        img_np = np.array(img)

        if img_np.ndim == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img_np)


class HistogramEqualizationTransform:
    """Global histogram equalization on the Y (luma) channel (YCrCb)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        img_np = np.array(img)

        if img_np.ndim == 3:
            ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            img_np = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        return Image.fromarray(img_np)


class GammaCorrectionTransform:
    """Gamma correction applied per RGB channel via a lookup table."""

    def __init__(self, gamma: float = 1.5):
        self.gamma = gamma
        self.table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]
        ).astype("uint8")

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        img_np = np.array(img)
        img_np = cv2.LUT(img_np, self.table)

        return Image.fromarray(img_np)


class BoundingBoxCropTransform:
    """
    Automatically crop to the bounding box of the largest foreground object
    (the larva) using Otsu thresholding and contour detection.

    Algorithm:
      1. Convert to grayscale and apply a small blur for threshold stability.
      2. Run Otsu threshold in both normal and inverted mode; pick the mode
         whose largest contour covers a bigger area (handles both dark-on-light
         and light-on-dark microscopy setups automatically).
      3. Morphologically close the mask to fill small holes.
      4. Take the single largest contour as the region of interest.
      5. Expand the bounding box by `padding` pixels on every side, clamped
         to image bounds.
      6. Reject the crop if the detected region is unreasonably small
         (< min_area_ratio of the image) or covers almost the whole image
         (> max_area_ratio -- threshold probably failed). Fall back to the
         original image in both cases.

    This is applied BEFORE GaussianBlurTransform and Resize so that
    InceptionV3 receives a tightly focused view of the larva.
    """

    def __init__(
        self,
        padding: int = 10,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.95,
    ):
        self.padding = padding
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def _largest_contour_area(self, thresh):
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0, None
        largest = max(contours, key=cv2.contourArea)
        return cv2.contourArea(largest), largest

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]
        image_area = h * w

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Slight blur for stable Otsu threshold
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try both polarities; keep whichever gives the larger largest contour
        _, thresh_inv = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, thresh_nrm = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
        thresh_nrm = cv2.morphologyEx(thresh_nrm, cv2.MORPH_CLOSE, kernel)

        area_inv, contour_inv = self._largest_contour_area(thresh_inv)
        area_nrm, contour_nrm = self._largest_contour_area(thresh_nrm)

        contour = contour_inv if area_inv >= area_nrm else contour_nrm
        best_area = max(area_inv, area_nrm)

        # Validity checks: skip crop if detection looks wrong
        if (
            contour is None
            or best_area < self.min_area_ratio * image_area
            or best_area > self.max_area_ratio * image_area
        ):
            return img  # fall back to full image

        x, y, bw, bh = cv2.boundingRect(contour)
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(w, x + bw + self.padding)
        y2 = min(h, y + bh + self.padding)

        cropped = img_np[y1:y2, x1:x2]

        # Safety: don't return a degenerate crop
        if cropped.size == 0:
            return img

        return Image.fromarray(cropped)


class GaussianBlurTransform:
    """
    Gaussian blur for noise reduction in microscopy images.

    Applied AFTER BoundingBoxCropTransform so blur acts on the focused
    larva region, not the full background. A small kernel (5×5, sigma=1)
    smooths sensor/optical noise without destroying fine morphological
    details like siphon shape or body segmentation that the classifier uses.

    kernel_size is forced to be odd (OpenCV requirement).
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        # Ensure kernel_size is odd
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        img_np = np.array(img)
        img_np = cv2.GaussianBlur(
            img_np, (self.kernel_size, self.kernel_size), self.sigma
        )
        return Image.fromarray(img_np)
