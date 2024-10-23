import os
import torch
from PIL import Image
from reflectool.models.base_model import Base_Model, LOCAL_MODEL_PATHS
from reflectool.models.conversations import get_conv
from reflectool.models.medpp_llava.model.builder import load_pretrained_model
from reflectool.models.medpp_llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from reflectool.models.medpp_llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class LLavaMedPPModel(Base_Model):
    def __init__(self, model_name, system_prompt, stops, cuda_id=0):
        super().__init__()
        self.device = f"cuda:{cuda_id}"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(LOCAL_MODEL_PATHS[model_name], model_base=None, model_name="llava-llama-med-8b-captioner", device_map=self.device, device=self.device)
        self.conv = get_conv("llavamedpp")
    
    def __call__(self, images: str):
        inputs = ""
        
        if self.model.config.mm_use_im_start_end:
            inputs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inputs
        else:
            inputs = DEFAULT_IMAGE_TOKEN + '\n' + inputs
    
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], inputs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(images).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)[0].unsqueeze(0).half().to(self.device)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True)
        
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
