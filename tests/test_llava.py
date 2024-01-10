import unittest
import torch
from transformers import LlamaConfig, GPTNeoXConfig
from robin.model import LlavaLlamaForCausalLM, LlavaGPTNeoXForCausalLM, LlavaMistralForCausalLM

@unittest.skip('Need 300 second to Run LlavaLlama')
class TestClipLlama(unittest.TestCase):
    def setUp(self):
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        config.vision_tower = "openai/clip-vit-large-patch14-336"
        config.mm_vision_tower = config.vision_tower
        config.mm_projector_type = "mlp2x_gelu"
        config.mm_vision_select_layer = -2
        config.mm_vision_select_feature = "patch"
        config.mm_hidden_size = 1024
        config.pretrain_mm_mlp_adapter = None

        self.model = LlavaLlamaForCausalLM(config)
        self.model.get_model().initialize_vision_modules(config)
        self.vision_tower = self.model.get_vision_tower()
        
        assert self.model is not None
        assert self.vision_tower is not None

    def test_llava_llama_forward(self):
        input_ids = torch.tensor([[101, 2054, 2003, 1037, 2518, 1012, 102]]) 
        use_cache = True 
        images = torch.randn(1, 3, 336, 336)

        output = self.model(
            input_ids=input_ids,
            use_cache=use_cache,
            images=images,
        )

        assert output[0].shape == torch.Size([1, 7, 32000])
        

@unittest.skip('Pass')
class TestClipPythia(unittest.TestCase):
    def setUp(self):
        config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-410m")
        config.vision_tower = "openai/clip-vit-large-patch14-336"
        config.mm_vision_tower = config.vision_tower
        config.mm_projector_type = "mlp2x_gelu"
        config.mm_vision_select_layer = -2
        config.mm_vision_select_feature = "patch"
        config.mm_hidden_size = 1024
        config.pretrain_mm_mlp_adapter = None

        self.model = LlavaGPTNeoXForCausalLM(config)
        self.model.get_model().initialize_vision_modules(config)
        self.vision_tower = self.model.get_vision_tower()
        
        assert self.model is not None
        assert self.vision_tower is not None

    def test_llava_neox_forward(self):
        input_ids = torch.tensor([[101, 2054, 2003, 1037, 2518, 1012, 102]]) 
        use_cache = True 
        images = torch.randn(1, 3, 336, 336)
        output = self.model(
            input_ids=input_ids,
            use_cache=use_cache,
            images=images,
        )
        
        assert output[0].shape == torch.Size([1, 7, 50304])

    def test_llava_neox_from_pretrained(self):
        self.pretrain_model = LlavaGPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m")


class TestOpenClipPythia(unittest.TestCase):
    def setUp(self):
        config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-410m")
        config.vision_tower = "ViT-B-16/laion2b_s34b_b88k"
        config.mm_vision_tower = config.vision_tower
        config.mm_projector_type = "mlp2x_gelu"
        config.mm_vision_select_layer = -1
        config.mm_vision_select_feature = "patch"
        config.mm_hidden_size = 768
        config.pretrain_mm_mlp_adapter = None

        self.model = LlavaGPTNeoXForCausalLM(config)
        self.model.get_model().initialize_vision_modules(config)
        self.vision_tower = self.model.get_vision_tower()
        
        assert self.model is not None
        assert self.vision_tower is not None

    def test_llava_neox_forward(self):
        input_ids = torch.tensor([[101, 2054, 2003, 1037, 2518, 1012, 102]]) 
        use_cache = True 
        images = torch.randn(1, 3, 224, 224)
        output = self.model(
            input_ids=input_ids,
            use_cache=use_cache,
            images=images,
        )
        
        assert output[0].shape == torch.Size([1, 7, 50304])


if __name__ == '__main__':
    unittest.main()