from reflectool.models.base_model import *
from reflectool.models.openai_model import *
from reflectool.models.huatuo_vision_model import HuatuoVision_Model
from reflectool.models.MiniCPM import MiniCPM_Model
from reflectool.models.InternVLChat import InternVLChatModel
from reflectool.models.llava_med_pp import LLavaMedPPModel

def get_model(model_name, system_prompt=None, stops=[], vllm_serve=False, vllm_serve_url=None, cuda_id=0):
    # api model
    if model_name == "gpt-3.5-turbo":
        return OpenAI_Model(model_type="gpt-3.5-turbo", system_prompt=system_prompt, stops=stops)
    
    elif model_name == "gpt-4o":
        return OpenAI_Model(model_type="gpt-4o", system_prompt=system_prompt, stops=stops)
    
    elif model_name == "gpt-4o-mini":
        return OpenAI_Model(model_type="gpt-4o-mini", system_prompt=system_prompt, stops=stops)

    elif model_name == "internvl-chat-v1.5":
        return InternVLChatModel(model_name, system_prompt=system_prompt, stops=stops)

    elif model_name == "huatuo-vision-7b" or model_name == "huatuo-vision-34b":
        return HuatuoVision_Model(model_name, system_prompt=system_prompt, stops=stops, cuda_id=cuda_id)
    
    elif model_name == "minicpm-v-2.6":
        return MiniCPM_Model(model_name, system_prompt=system_prompt, stops=stops, cuda_id=cuda_id)
    
    elif model_name == "llavamedpp":
        return LLavaMedPPModel(model_name, system_prompt=system_prompt, stops=stops, cuda_id=cuda_id)
    
    elif model_name in ["qwen2-72b-int4", "llama3.1-70b-int4", "qwen2-7b", "llama3.1-8b", "llama3-8b"]:
        try:
            if vllm_serve and vllm_serve_url is not None:
                from models.vllm_serve_models import VLLMServeModel
                return VLLMServeModel(model_name, vllm_serve_url=vllm_serve_url, system_prompt=system_prompt, stops=stops)
            else:
                from models.vllm_models import VLLM_Model
                return VLLM_Model(model_name, system_prompt=system_prompt, stops=stops)
        except:
            return Local_Model(model_name, system_prompt=system_prompt, stops=stops)

    else:
        return Local_Model(model_name, system_prompt=system_prompt, stops=stops)