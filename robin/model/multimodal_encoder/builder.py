from .clip_encoder import CLIPVisionTower
from .open_clip import OpenCLIPVisionTower
from .timm_vision import TimmVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower_type = getattr(vision_tower_cfg, 'vision_tower_type', None)

    if vision_tower_type == 'open_clip':
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower_type == 'clip':
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower_type == 'timm':
        return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        print(f"""[Warning] Unknown vision tower type: {vision_tower_type} Use open_clip as default vision tower now.
              """)
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
