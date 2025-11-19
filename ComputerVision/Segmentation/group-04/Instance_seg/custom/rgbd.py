import os
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadRGBDFromFile(BaseTransform):
    def __init__(self, depth_root, npy_suffix='.npy', to_float32=True, depth_max_meter=80.0):
        self.depth_root = depth_root
        self.npy_suffix = npy_suffix
        self.to_float32 = to_float32
        self.depth_max_meter = depth_max_meter

    def transform(self, results: dict) -> dict:
        rgb_path = results['img_path']
        rel = results.get('ori_filename', os.path.basename(rgb_path))
        base, _ = os.path.splitext(rel)
        npy_path = os.path.join(self.depth_root, base + self.npy_suffix)

        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"No depth file for {rgb_path}")

        # 读 RGB
        img = mmcv.imread(rgb_path, flag='color')

        # 读 Depth .npy
        depth = np.load(npy_path)
        if depth.ndim == 3:
            depth = depth[..., 0]

        if depth.shape[:2] != img.shape[:2]:
            depth = mmcv.imresize(depth, (img.shape[1], img.shape[0]), interpolation='nearest')

        if self.to_float32:
            img = img.astype(np.float32)
            depth = depth.astype(np.float32)

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0.0, self.depth_max_meter)
        depth = depth / self.depth_max_meter
        depth = depth[:, :, None]

        img4 = np.concatenate([img, depth], axis=2)
        h, w = img4.shape[:2]
        results['img'] = img4
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)
        results['pad_shape'] = (h, w)
        results['num_channels'] = 4
        results['img_fields'] = ['img']
        results.setdefault('ori_filename', os.path.basename(rgb_path))
        results['img_path_rgb'] = rgb_path
        results['img_path_depth'] = npy_path
        return results
