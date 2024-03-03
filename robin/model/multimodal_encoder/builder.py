import os
from .clip_encoder import CLIPVisionTower
from .open_clip import OpenCLIPVisionTower
from .timm_vision import TimmVisionTower
from .encoder_info import CLIP_COMPATIBLE_MODELS, OPENCLIP_CONFIG_MAP

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if os.path.exists(vision_tower):
        name = vision_tower.split("/")[-1].lower()
        name = name.replace("hf-hub:", "")

        if name in CLIP_COMPATIBLE_MODELS:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "dino" in str(vision_tower).lower():
            return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif name in OPENCLIP_CONFIG_MAP.keys():
            return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            print(f'Local model not handled in configs (might crash): {vision_tower}')
            return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        vision_tower_tmp = vision_tower.lower()
        if vision_tower_tmp.startswith("hf-hub:"):
            vision_tower_tmp = vision_tower_tmp.replace("hf-hub:", "")

        if vision_tower_tmp.startswith("openai") or vision_tower_tmp.startswith("laion") or vision_tower_tmp.startswith("facebook"):
            vision_tower = vision_tower.replace("hf-hub:", "")
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "dino" in vision_tower_tmp:
            return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
