import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import open_clip
from torch import nn
import os

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
            pass
        
        # self.hidden_size = 1152 if "SO400M" in self.vision_tower_name else 768  # TODO: this is a hack, we should read this from config.embed_dim

        self.device = None
        self.dtype = None
        
    def load_model(self):
        
        if os.path.exists(self.vision_tower_name):
            name = self.vision_tower_name.split("/")[-1]

            map_name_config = {
                'eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k': 'EVA02-E-14-plus',
                'ViT-SO400M-14-SigLIP-384': 'ViT-SO400M-14-SigLIP-384',
                'ViT-bigG-14-CLIPA-336-datacomp1B': 'ViT-bigG-14-CLIPA-336',
                'ViT-bigG-14-CLIPA-datacomp1B': 'ViT-bigG-14-CLIPA',
                'DFN5B-CLIP-ViT-H-14': 'ViT-H-14' 
            }

            config = map_name_config[name] if name in map_name_config.keys() else name

            self.vision_tower, self.image_processor = create_model_from_pretrained(config, pretrained=self.vision_tower_name+'/open_clip_pytorch_model.bin')
        else:
            self.vision_tower, self.image_processor = create_model_from_pretrained(self.vision_tower_name)
        

        
        self.vision_tower = self.vision_tower.visual

        if "timm" in self.vision_tower_name or "eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k" in self.vision_tower_name or "ViT-SO400M-14-SigLIP-384" in self.vision_tower_name:
            self.hidden_size = self.vision_tower.trunk.embed_dim
        else:
            # self.hidden_size = self.vision_tower.num_features
            # self.hidden_size = self.vision_tower.output_dim
            self.hidden_size = self.vision_tower.proj.data.shape[0]
            self.vision_tower.pshape = self.vision_tower.proj.data.shape
            self.vision_tower.proj = None
            self.vision_tower.output_tokens = True
        
        # self.vision_tower.trunk.output_tokens = True
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        
        
        
        
    def feature_select(self, image_forward_outs):
        
        assert False, ("not implemented")
        
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images): # shape, CLS shape, image tokens shape, and hidden size
        
        if type(images) is list:
            image_features = []
            for image in images:
                cls_token, image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature)
        else:#This should always be unsqueezed, if we have multiple items just stack them before this
            print('==========================')
            print('pshape', self.vision_tower.pshape)
            print('hidden_size', self.hidden_size)
            print('images.shape', images.shape)
            cls_token, image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            print('cls_token.shape', cls_token.shape)
            print('image_features.shape', image_features.shape)
        cls_token = cls_token.unsqueeze(1)
        image_features = torch.cat((cls_token, image_features), dim=1)
        return image_features

    @property
    def dummy_feature(self):#We want to get back dummy featrues for whatever reason... we need to know the output shape.
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return next(self.vision_tower.parameters()).dtype

    # @property
    # def device(self):
    #     return next(self.vision_tower.parameters()).device

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
