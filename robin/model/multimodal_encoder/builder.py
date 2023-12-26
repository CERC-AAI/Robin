import os
from .clip_encoder import CLIPVisionTower
from .open_clip import OpenCLIPVisionTower
from .timm_vision import TimmVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if os.path.exists(vision_tower):
        print('===> ' + vision_tower + ' <===')
        name = vision_tower.split("/")[-1].lower()
        clip_compatible = ['metaclip-h14-fullcc2.5b', 'clip-vit-bigg-14-laion2b-39b-b160k', 'metaclip-l14-fullcc2.5b', 'clip-vit-large-patch14-336']

        if name in clip_compatible:
            print('===> ===> CLIPVisionTower')
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "dino" in str(vision_tower).lower():
            print('===> ===> TimmVisionTower')
            return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            print('===> ===> OpenCLIPVisionTower')
            return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        if vision_tower.lower().startswith("openai") or vision_tower.lower().startswith("laion"):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "dino" in str(vision_tower).lower():
            return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
