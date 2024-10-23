import os
import re
import torch
from PIL import Image
from reflectool.models.base_model import Base_Model, disable_torch_init, LOCAL_MODEL_PATHS
from transformers import AutoTokenizer, AutoModel


class MiniCPM_Model(Base_Model):
    def __init__(self, model_name, system_prompt, stops, cuda_id=0):
        super().__init__()

        model_path = LOCAL_MODEL_PATHS[model_name.lower()]
        self.model_path = os.path.expanduser(model_path)
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, 
                                            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        
        self.model.eval().cuda(cuda_id)
    
    def __call__(self, inputs, images = None):
        if images is not None:
            image = Image.open(images).convert('RGB')
            msgs = [{'role': 'user', 'content': [image, inputs]}]
        else:
            msgs = [{'role': 'user', 'content': [inputs]}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        return res