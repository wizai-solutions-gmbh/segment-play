
from typing import Any, Optional

import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry

from segmentation.base import Segmentation


class MobileSam(Segmentation):
    def __init__(self) -> None:
        super().__init__()
        checkpoint = './models/mobile_sam.pt'
        model_type = 'vit_t'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)  # type: ignore
        self.predictor: SamPredictor = SamPredictor(sam)

    def get_image_embedding(self) -> Any:
        return self.predictor.get_image_embedding().cpu().numpy()

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)
        self.image_embedding = self.get_image_embedding()

    def bbox_masks(
        self,
        bb: np.ndarray,
        points: Optional[np.ndarray] = None,
        point_modes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        point_coords = None
        point_labels = None
        if points is not None and points.any():
            labels = np.ones(points.shape[0])
            if point_modes is not None:
                labels = point_modes
            point_labels = labels
            point_coords = points

        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bb[None, :],
            multimask_output=False,
        )
        masks = masks > self.predictor.model.mask_threshold
        return masks
