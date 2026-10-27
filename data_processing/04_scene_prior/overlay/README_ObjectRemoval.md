## ObjectRemoval

Install (editable) from the repo root:

```bash
bash data_processing/04_scene_prior/setup_omnipaint.sh
# (clones OmniPaint and runs: pip install -e data_processing/04_scene_prior/OmniPaint)
```

### Usage

```python
import cv2
from ObjectRemoval import ObjectRemover, ObjectRemovalConfig

# `input_is_bgr=True` if you pass OpenCV images (cv2.imread gives BGR)
remover = ObjectRemover(ObjectRemovalConfig(input_is_bgr=True, device="cuda:0"))

img_bgr = cv2.imread("/path/to/image.jpg")
mask = cv2.imread("/path/to/mask.png", cv2.IMREAD_GRAYSCALE)  # white=remove, black=keep

out_pil = remover.remove(img_bgr, mask)
out_pil.save("/path/to/edited.jpg")
```

