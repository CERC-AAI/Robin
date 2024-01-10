import unittest
import torch
import torch.nn as nn
from robin.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from robin.model.multimodal_encoder.open_clip import OpenCLIPVisionTower
from dataclasses import dataclass


@dataclass
class Args:
    mm_vision_select_layer: int = 0
    mm_vision_select_feature: str = 'patch'

@unittest.skip('Pass')
class TestClip(unittest.TestCase):
    def setUp(self):
        vision_tower = 'openai/clip-vit-large-patch14-336'
        args = Args()
        args.mm_vision_select_layer = -2
        args.mm_vision_select_feature = 'patch'
        self.model = CLIPVisionTower(vision_tower, args, False)
    
    def test_patch_forward(self):
        images = torch.randn(1, 3, 336, 336)
        image_features = self.model(images)
        assert image_features.shape[1] == 576 # (336 // 24) ** 2
        assert image_features.shape[2] == 1024


class TestOpenClip(unittest.TestCase):
    def setUp(self):
        self.vision_tower = 'ViT-B-16/laion2b_s34b_b88k'
        self.args = Args()
        self.args.mm_vision_select_layer = -1
        self.args.mm_vision_select_feature = 'patch'
        self.model = OpenCLIPVisionTower(self.vision_tower, self.args, False)

    def test_patch_forward(self):
        images = torch.randn(1, 3, 224, 224)
        image_features = self.model(images)
        assert image_features.shape[1] == 196 # (224 // 16) ** 2
        assert image_features.shape[2] == 768

    def test_cls_patch_forward(self):
        self.args.mm_vision_select_feature = 'cls_patch'
        self.model = OpenCLIPVisionTower(self.vision_tower, self.args, False)
        images = torch.randn(1, 3, 224, 224)
        image_features = self.model(images)
        assert image_features.shape[1] == 197 # (224 // 16) ** 2 + 1
        assert image_features.shape[2] == 768


if __name__ == '__main__':
    unittest.main()