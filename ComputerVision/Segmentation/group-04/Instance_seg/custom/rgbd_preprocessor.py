import torch
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor

@MODELS.register_module()
class RGBDDetDataPreprocessor(DetDataPreprocessor):
    def __init__(self, mean, std, **kwargs):
        super().__init__(mean=[0.0], std=[1.0], **kwargs)

        # 存 4 通道 mean/std
        m = torch.tensor(mean, dtype=torch.float32)
        s = torch.tensor(std, dtype=torch.float32)
        self.register_buffer('_mean', m.view(-1, 1, 1), persistent=False)
        self.register_buffer('_std', s.view(-1, 1, 1), persistent=False)

    def stack_batch(self, data: list, samples_per_gpu: int = 1) -> dict:
        imgs = []
        for sample in data:
            img = sample['inputs']
            img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            # 归一化
            if img.size(0) == 4:
                depth = img[3:4]
                if depth.max() > 0:
                    depth = depth / (depth.max() + 1e-6)
                img = torch.cat([img[:3], depth], dim=0)

            img = (img - self._mean) / self._std
            imgs.append(img)

        imgs = self.pad(imgs)
        batch_inputs = torch.stack(imgs, dim=0)
        return dict(inputs=batch_inputs)

