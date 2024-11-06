import os
import torch
from vllm import LLM, SamplingParams

from reflectool.models.base_model import Base_Model, disable_torch_init
from reflectool.models.base_model import LOCAL_MODEL_PATHS
from reflectool.models.conversations import get_conv, SeparatorStyle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from openai import OpenAI

# URL_MAPPING = {
#     # "qwen2-7b": "http://SH-IDC1-10-140-0-173:10002/v1",
#     "qwen2-72b-int4": "http://SH-IDC1-10-140-0-173:10002/v1"
# }

class VLLMServeModel(Base_Model):
    def __init__(self, model_name, vllm_serve_url, system_prompt, stops):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name.lower()]
        self.model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

        self.client = OpenAI(
            base_url=f"{vllm_serve_url}/v1"
        )

        self.system_prompt = system_prompt
        self.conv = get_conv(model_name)
        if self.system_prompt is not None:
            self.conv.system = self.system_prompt

        # format stop token id
        self.stop_ids = []
        self.__add_stop_token_id__(stops)
        self.__add_stop_token_id__(self.tokenizer.eos_token_id)
        self.__add_stop_token_id__(self.conv.stop_token_ids)
        self.__add_stop_token_id__(self.conv.stop_str)
        self.__add_stop_token_id__([self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2])
        self.stop_ids = [stop_id for stop_id in self.stop_ids]

        self.__get_stop_token__()
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1024, stop=self.stop_strs)
    
    def __get_stop_token__(self):
        self.stop_strs = [self.tokenizer.decode(stop_id) for stop_id in self.stop_ids]
    
    def __add_stop_token_id__(self, stops):
        if stops is None:
            return 

        if not isinstance(stops, list):
            stops = [stops]

        # add stop tokens
        for stop_item in stops:
            if isinstance(stop_item, str):
                stop_id = self.tokenizer(stop_item, add_special_tokens=False).input_ids
            elif isinstance(stop_item, int):
                stop_id = [stop_item]
            else:
                continue
            
            if stop_id not in self.stop_ids:
                self.stop_ids.append(stop_id)
    
    def __get_prompt__(self, inputs, system_prompt=None):
        conv = self.conv.copy()
        if system_prompt is not None:
            conv.system = system_prompt

        conv.append_message(conv.roles[0], inputs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt

    def __call__(self, inputs, system_prompt=None, n=1, temperature=0.0, use_beam_search=False):
        inputs = self.__get_prompt__(inputs, system_prompt)
        completion = self.client.completions.create(
            model=self.model_path,
            prompt=inputs,
            max_tokens=1024,
            stop=self.stop_strs
        )

        response = completion.choices[0].text
        return self.postprocessed(response)

