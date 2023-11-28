import argparse
from io import BytesIO

from PIL import Image
from PIL import Image
import requests
import torch
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, process_images_easy, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
class RobinLavaPipeline:
    
    def __init__(self, model_path, model_base, device, image_file, load_8bit = False, load_4bit = False, temperature = .2, max_new_tokens = 512):
        
        self.model_path = model_path
        self.model_base = model_base
        self.device = device
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        #Sure why not
        disable_torch_init()
    
        model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, image_processor, context_len = load_pretrained_model(self.model_path, self.model_base, model_name, self.load_8bit, self.load_4bit, device=self.device)
        
        self.conv = conv_templates['vicuna_v1'].copy()
        self.roles = self.conv.roles

        image = load_image(image_file)

        # Similar operation in model_worker.py
        
        image_tensor = process_images_easy([image], image_processor, "pad")
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        self.image = image
        self.image_tensor = image_tensor

    
    def __call__(self, messages):
        
        for message in messages:
            role = message["role"]
            if role != "USER" and role != "ASSISTANT":
                print("Only USER and ASSISTANT roles are supported, exiting")
                exit()
            content = message["content"]
        
        
        #First message
        inp = messages[0]["content"]
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(messages[0]["role"], inp)
        self.image = None
        messages.pop(0)
        #We typically assume that follows the format of User, then assistant.
        for message in messages:
            self.conv.append_message(message["role"], message["content"])
            
        #At the very end, we expect to see a user, so we add the empty assistant.
        self.conv.append_message(self.conv.roles[1], None)
            
            
        prompt = self.conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
        #TODO 
        blockPrint()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.image_tensor,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        enablePrint()
        
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs
        return self.conv.messages


