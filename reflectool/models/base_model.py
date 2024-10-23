import os
import pdb
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
# from transformers import LlavaMistralForCausalLM
from reflectool.models.conversations import conv_templates
from reflectool.models.conversations import get_conv, SeparatorStyle

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, 'config.yaml'), 'r') as f:
    # 将YAML内容转换为字典
    LOCAL_MODEL_PATHS = yaml.safe_load(f)["model"]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class Base_Model:
    def __init__(self):
        pass

    def postprocessed(self, outputs):
        return outputs.strip(" \n")

    def __call__(self):
        raise NotImplementedError

    def multiple_choice_selection(self):
        raise NotImplementedError

    def get_logit_bias(self, state_num=4):
        raise NotImplementedError

class API_Model(Base_Model):
    def __init__(self, api_key, stops):
        super().__init__()
        self.api_key = api_key
        self.stops = stops
    
    def __call__(self):
        return super().__call__()

class Local_Model(Base_Model):
    def __init__(self, model_name, system_prompt, stops):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name.lower()]
        self.model_path = os.path.expanduser(model_path)
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.model.eval()

        self.system_prompt = system_prompt
        self.conv = get_conv(model_name)
        self.conv.system = self.system_prompt

        # format stop token id
        self.stop_ids = []
        self.__add_stop_token_id__(stops)
        self.__add_stop_token_id__([self.tokenizer.eos_token_id])
        self.__add_stop_token_id__(self.conv.stop_token_ids)
        self.__add_stop_token_id__([self.conv.stop_str])
        self.__add_stop_token_id__([self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2])
        self.stop_ids = [torch.tensor(stop_id).cuda() for stop_id in self.stop_ids]

        self.__get_stop_token__()
        self.stop_criteria = KeywordsStoppingCriteria(self.stop_ids)
    
    def __get_stop_token__(self):
        self.stop_strs = [self.tokenizer.decode(stop_id) for stop_id in self.stop_ids]
    
    def __add_stop_token_id__(self, stops):
        # add stop tokens
        if stops is None:
            return 
        
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
    
    def get_logit_bias(self, state_num=4):
        state_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        logit_bias = {}
        # pdb.set_trace()
        for i in range(state_num):
            logit_bias[self.tokenizer(state_list[i], add_special_tokens=False)["input_ids"][0]] = 100

        return logit_bias
    
    def __call__(self, inputs, system_prompt=None, n=1, temperature=0.0, use_beam_search=False):
        inputs = self.__get_prompt__(inputs, system_prompt)
        inputs = self.tokenizer([inputs])

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=torch.as_tensor(inputs.input_ids).cuda(),
                num_beams=n,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                max_new_tokens=500,
                stopping_criteria=StoppingCriteriaList([self.stop_criteria]))
     
        final_outputs = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self.tokenizer.decode(final_outputs)

        return self.postprocessed(outputs)
    
    def postprocessed(self, outputs):
        for stop_str in self.stop_strs:
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
        return super().postprocessed(outputs)

    def multiple_choice_selection(self, inputs, logit_bias):
        logits_processor_list = LogitsProcessorList([
            LogitBiasLogitsProcessor(logit_bias),
        ])

        inputs = self.tokenizer([inputs])
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=False,
                max_new_tokens=1,
                logits_processor=logits_processor_list,
            )
        
        final_outputs = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self.tokenizer.decode(final_outputs)

        return outputs

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for keyword in self.keywords:
            if torch.all(keyword == input_ids[0][-len(keyword):]).item():
                return True
        return False

class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, logit_bias):
        self.logit_bias = logit_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        for index in self.logit_bias.keys():
            scores[:, index] += self.logit_bias[index]
        return scores


