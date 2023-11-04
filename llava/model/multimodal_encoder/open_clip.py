import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from torch import nn

class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            print("not implemented")
            exit()
            # self.cfg_only = OpenCLIPVisionConfig.from_pretrained(self.vision_tower_name)
        
        self.hidden_size = 768
        
    def load_model(self):
        
        #So we need to run this code from OUTSIDE this code once before we can do this. I don't know why either.
        self.vision_tower, self.image_processor = create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP", cache_dir = "./")
        self.vision_tower = self.vision_tower.visual
        
        #Need to make sure it has the attribute. If not, uhoh. Might need to modify code this also, depending.
        self.vision_tower.trunk.output_tokens = True
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        
        
        
        
    def feature_select(self, image_forward_outs):
        
        print("not implemented")
        exit()
        
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            print("Why are you a list right now")
            exit()
        else:#This should always be unsqueezed, if we have multiple items just stack them before this
            image_features = model.encode_image(image)
        return image_features

    @property
    def dummy_feature(self):#We want to get back dummy featrues for whatever reason... we need to know the output shape.
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
        

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    def hidden_size(self):
        return self.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
