import argparse
from io import BytesIO

from PIL import Image
from PIL import Image
import requests
import torch
from transformers import TextStreamer

from robin.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robin.conversation import conv_templates, SeparatorStyle
from robin.mm_utils import process_images, process_images_easy, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from robin.model.builder import load_pretrained_model
from robin.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

import sys, os



class LlavaMistralPipeline:
    
    def __init__(self, model_path, model_base, device="cuda", load_8bit=False, load_4bit=False, temperature=.2, max_new_tokens=512):
        
        self.model_path = model_path
        self.model_base = model_base
        self.device = device
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # TODO: Simon: make this work reliably
        if load_4bit or load_8bit:
            print("WARNING: 4bit or 8bit models might not work as expected")

        # Sure why not
        disable_torch_init()
    
        model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.model_path, self.model_base, model_name, self.load_8bit, self.load_4bit, device=self.device)
        

    def _load_image_tensor(self, image_file):
        image = load_image(image_file)

        # Similar operation in model_worker.py
        image_tensor = process_images_easy([image], self.image_processor, "pad")
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        return image_tensor

    
    def __call__(self, messages):
        conv = conv_templates['vicuna_v1'].copy()
        assert conv.roles == ('USER', 'ASSISTANT')

        # First message
        assert messages[0]["role"] == "USER"
        inp = messages[0]["content"]
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        # TODO: Simon: handle no image case
        assert "image" in messages[0], 'First message needs to have an image url'
        image_tensor = self._load_image_tensor(messages[0]["image"])

        conv.append_message("USER", inp)


        # Remaining messages
        # We typically assume that follows the format of user, then assistant.
        for message in messages[1:]:
            assert message["role"] in ["USER", "ASSISTANT"], f"Only USER and ASSISTANT roles are supported, got {message['role']}"
            assert "image" not in message, "Images can only be in the first user message"
            conv.append_message(message["role"], message["content"])
            
        # At the very end, we expect to see a user, so we add the empty assistant.
        conv.append_message("ASSISTANT", None)
            
            
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # For vicuna_v1, stop_str == "</s>"
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # conv.messages[-1][-1] = outputs

        return [*messages, {"role": "ASSISTANT", "content": outputs}]
