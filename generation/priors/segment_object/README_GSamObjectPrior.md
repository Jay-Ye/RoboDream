## GSamObjectPrior

Install (editable) from the repo root:

```bash
pip install -e generation/priors/segment_object
```

Usage:

```python
import cv2
from GSamObjectPrior import GroundedSamScenePrior, GSamScenePriorConfig

cfg = GSamScenePriorConfig(input_is_bgr=True)
prior = GroundedSamScenePrior(cfg)

img_bgr = cv2.imread("/path/to/image.jpg")
out = prior(img_bgr, ["a mug", "a bowl"])

scene_prior_object_img_bgr = out.scene_prior_object_img_bgr
merged_mask = out.merged_mask
masks = out.mask_list
labels = out.mask_labels
```

