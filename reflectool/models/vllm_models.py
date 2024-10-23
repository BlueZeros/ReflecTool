import os
import torch
from vllm import LLM, SamplingParams

from reflectool.models.base_model import Base_Model, disable_torch_init
from reflectool.models.local_model_path import LOCAL_MODEL_PATHS
from reflectool.models.conversations import get_conv, SeparatorStyle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList

class VLLM_Model(Base_Model):
    def __init__(self, model_name, system_prompt, stops):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name.lower()]
        self.model_path = os.path.expanduser(model_path)
        disable_torch_init()
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = LLM(model=self.model_path, trust_remote_code=True, max_model_len=min(32000, model_config.max_position_embeddings), max_seq_len_to_capture=8000, gpu_memory_utilization=0.9) # gpu_memory_utilization=0.8

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
        self.stop_ids = [torch.tensor(stop_id).cuda() for stop_id in self.stop_ids]

        self.__get_stop_token__()
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048, stop=self.stop_strs)
    
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
        sampling_params = SamplingParams(n=n, temperature=temperature if not use_beam_search else 0.0, use_beam_search=use_beam_search if n > 1 else False, max_tokens=2048, stop=self.stop_strs)

        inputs = self.__get_prompt__(inputs, system_prompt)
        outputs = self.model.generate(prompts=inputs, sampling_params=sampling_params)

        if n == 1:
            outputs = outputs[0].outputs[0].text.strip()
            return self.postprocessed(outputs)
        else:
            outputs = [self.postprocessed(output.text.strip()) for output in outputs[0].outputs]
            return outputs