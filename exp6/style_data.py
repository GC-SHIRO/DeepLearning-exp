from pathlib import Path
import hashlib
import numpy as np
import requests
from PIL import Image, ImageOps, ImageFilter

# 由暗到亮的六种印刷色，供配对训练和 LoRA 风格训练共用。
PALETTE = np.array([(20,36,53),(40,81,96),(74,136,139),(209,135,82),(241,192,126),(250,232,187)], dtype=np.uint8)

def photo(row, root, size):
    filename = row.image_id + '.jpg'
    path = root.parent / 'exp4' / 'data' / 'coco_cn' / 'images' / filename
    if not path.exists():
        path = root / 'data' / filename
        if not path.exists():
            response = requests.get(row.url, timeout=60)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != row.sha256:
                raise ValueError(f'下载图像与清单不一致: {filename}')
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(response.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
        raise ValueError(f'图像与清单不一致: {filename}')
    with Image.open(path) as image:
        return ImageOps.fit(image.convert('RGB'), (size, size), method=Image.Resampling.LANCZOS)

def poster(image):
    gray = np.asarray(ImageOps.grayscale(image.filter(ImageFilter.MedianFilter(5))))
    return Image.fromarray(PALETTE[np.digitize(gray, [42,85,128,171,214])])
