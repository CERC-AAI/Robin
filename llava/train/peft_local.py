import torch
from torch import nn
import loralib as lora

def convert_model(model):
    
    #Basically we need to apply lore to only the stuff
    for name, child, in model.named_children():
        
        if isinstance(child, nn.Linear):
            
            new = lora.Linear(child.in_features, child.out_features, r = 16, dtype = torch.float32)
            new.weight = child.weight
            new.bias = child.bias
            setattr(model, name, new)
        elif isinstance(child, LlamaRMSNorm):#fix
            child.requires_grad = True
        else:
            convert_model(model)
    
    
    return model