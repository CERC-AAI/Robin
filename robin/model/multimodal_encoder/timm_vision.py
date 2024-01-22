import timm
import torch
from torch import nn


class TimmVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        assert args.mm_vision_select_layer == -1, "timm support output tokens of last layer only"
        if vision_tower == 'vit_so400m_patch14_siglip_384':
            # Siglip support patch model only, there seems no cls token
            assert args.mm_vision_select_feature == 'patch' 
            # Patch model will drop the first token of image feature
            # Siglip don't have cls one, therefore we use cls_patch model to convserve all patch tokens
            args.mm_vision_select_feature = 'cls_patch'

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.hidden_size = None # Place Holder get this value in load_model function
        self.dtype = None # Place Holder get this value in train function
        self.is_loaded = False
        if not delay_load:
            self.load_model()
        
    def load_model(self):
        # if os.path.exists(self.vision_tower_name):
        #     # TODO support local cache
        #     name = self.vision_tower_name.split("/")[-1]
        #     self.vision_tower = timm.create_model(
        #         name,
        #         num_classes=0,  # remove classifier nn.Linear
        #         checkpoint_path=self.vision_tower_name+'/pytorch_model.bin', 
        #     )
        # else:
        self.vision_tower = timm.create_model(
            self.vision_tower_name,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        data_config = timm.data.resolve_model_data_config(self.vision_tower)
        self.hidden_size = self.vision_tower.num_features
        self.image_processor = timm.data.create_transform(**data_config, is_training=False)
        self.is_loaded = True
        
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        """
        TODO this intermediate feature is experimential, we should use this once it's stable
        https://github.com/huggingface/pytorch-image-models/discussions/2068
        """
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype))
                # In timm, output of forward_feature is stack of ([cls, patch])
                image_feature = self.feature_select(image_forward_outs).to(image.dtype)
                image_features.append(image_feature)
        else:
            # This incl cls token
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        return image_features

    @property
    def dummy_feature(self):#We want to get back dummy featrues for whatever reason... we need to know the output shape.
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only
    
    @property
    def device(self):
        return next(self.vision_tower.parameters()).device
    
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
