import os
from .clip_encoder import CLIPVisionTower
from .open_clip import OpenCLIPVisionTower
# from .timm_vision import TimmVisionTower

OPEN_CLIP_MODELS = [
    'ViT-B-16/laion2b_s34b_b88k',
    'ViT-G-14/laion2B-s34B-b88K',
]

CLIP_MODELS = [
    'openai/clip-vit-large-patch14-336'
]

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower in OPEN_CLIP_MODELS:
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif vision_tower.lower().startswith("openai"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
