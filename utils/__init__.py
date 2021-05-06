import os
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff',
]
def is_img(fpath):
    _, suffix = os.path.splitext(fpath)
    return suffix in IMG_EXTENSIONS

def get_cls(pkg, name):
    import importlib
    pkg = importlib.import_module(f"{pkg}.{name}")
    return getattr(pkg, name)

def tensor2img(tensor, n):
    np_tensor = tensor[:n, ...].detach().cpu().numpy()
    np_tensor = np.transpose(np_tensor, (0,2,3,1))
    n = np_tensor.shape[0]
    if n > 1:
        img = np.vstack([_] for _ in np_tensor)
    else:
        img = np_tensor[0]
    return np.clip(img*255, 0, 255).astype(np.uint8)

def show_remain(t):
    value = []
    for div in [60, 60, 24]:
        value.append(t % div)
        t //= div
    desc = []
    for d, v in zip(['Day', 'Min', 'Sec'], value[::-1]):
        desc.append(f"{v}{d}")
    return ",".join(desc)
