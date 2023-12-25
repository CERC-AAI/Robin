from PIL import Image
from torch import nn
import torch
import open_clip
import timm
import torch.nn.functional as F


class TimmVisionTower(nn.Module):
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
        
        self.device = None
        self.dtype = None
        
    def load_model(self):
        
        # print("VISION TOWER:", self.vision_tower_name)
        
        self.vision_tower = timm.create_model(
            self.vision_tower_name,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        
        data_config = timm.data.resolve_model_data_config(self.vision_tower)
        
        self.hidden_size = self.vision_tower.num_features#Can also use embed_dim
        
        #might need to change n shit. might not.
        transforms = timm.data.create_transform(**data_config, is_training=True)
        
        self.image_processor = transforms
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
    def forward(self, images):
        
        """
        output = model.forward_features(transforms(img).unsqueeze(0))
        # output is unpooled, a (1, 1370, 1024) shaped tensor
        
        output = model.forward_head(output, pre_logits=True)
        """
        if type(images) is list:
            image_features = []
            for image in images:
                cls_token, image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_forward_out)
        else:#This should always be unsqueezed, if we have multiple items just stack them before this
            cls_token, image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
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
