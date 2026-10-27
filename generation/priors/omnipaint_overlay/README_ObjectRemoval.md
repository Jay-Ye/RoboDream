## ObjectRemoval

Install from the repo root (clones OmniPaint, adds this wrapper and installs it in editable mode):

```bash
bash generation/setup_generation_env.sh
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

