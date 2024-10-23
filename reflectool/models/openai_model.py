import time
import os
import pdb
import json
import openai
from datetime import datetime
from openai import OpenAI
from reflectool.models import API_Model
import pytz
import base64

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class OpenAI_Model(API_Model):
    def __init__(self, 
                 model_type="gpt-3.5-turbo",
                 api_key=os.getenv("OPENAI_API_KEY"),
                 system_prompt=None,
                 stops=[]):
        super().__init__(api_key, stops)
        # self.t_start = time.perf_counter()
        self.model_type = model_type
        self.client = OpenAI(
            api_key=api_key
        )

        self.max_tokens = 300
        self.system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant. Please directly answer the question below."
    
    def get_logit_bias(self, state_num=4):
        return {(32+i):100 for i in range(state_num)}
    
    def log(self, message=None):
        self.cost_log["message"] = message
        with open(self.log_file, "w") as f:
            json.dump(self.cost_log, f, indent=4, ensure_ascii=False)

    def get_time(self):
        # 设置时区为中国时间
        china_timezone = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(china_timezone)
        return f"{current_time.year}-{current_time.month}-{current_time.day} {current_time.hour}:{current_time.hour}:{current_time.minute}:{current_time.second}"

    def update_log(self, message):
        self.cost_log["input_tokens"] += message.usage.prompt_tokens
        self.cost_log["output_tokens"] += message.usage.completion_tokens
        if self.model_type == "gpt-3.5-turbo":
            self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 1e-6 + self.cost_log["output_tokens"] * 2e-6
        elif self.model_type == "gpt-4o": 
            self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 1e-5 + self.cost_log["output_tokens"] * 3e-5
        self.cost_log["time_end"] = self.get_time()
    
    def __call__(self, inputs, images=None, system_prompt=None):
        if images is None:
            if system_prompt is not None:
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": inputs}
                ]
            else:
                message = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": inputs}
                ]
        else:
            base64_image = encode_image(images)
            if system_prompt is not None:
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": inputs}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ]
            else:
                message = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": inputs}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ]

        while True:
            try:
                # pdb.set_trace()
                # client = OpenAI(api_key=self.api_key)
                completion = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=message,
                    temperature=0,
                    seed=0,
                    max_tokens=self.max_tokens,
                    stop=self.stops,
                ) 
                outputs = completion.choices[0].message.content
                # self.update_log(completion)
                if outputs:
                    break 
                else:
                    print("Output is none, Retrying...")
            except openai.BadRequestError as e:
                if "You uploaded an unsupported image" in str(e):
                    return "The uploaded image is unsupported."
                else:
                    continue

            except openai.RateLimitError as e:
                print(e)
                t_rest = 60 - ( (time.perf_counter() - self.t_start) % 60 )
                print(f"surpass the tpm limits, wait for {t_rest} seconds...")
                time.sleep(t_rest)
                self.t_start = time.perf_counter()
            except openai.APITimeoutError as e:
                print("Timeout Error, Retrying...")
            except openai.APIConnectionError as e:
                print("Connect Error, Retrying...")
         
        return outputs
    
    def multiple_choice_selection(self, inputs, logit_bias):
        message = [{"role": "user", "content": inputs}]

        while True:
            try:
                # pdb.set_trace()
                # client = OpenAI(api_key=self.api_key)
                completion = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=message,
                    logit_bias=logit_bias,
                    temperature=0.0,
                    seed=0,
                    max_tokens=1,
                    )   
                self.update_log(completion)
                break 
            except openai.RateLimitError as e:
                print(e)
                t_rest = 60 - ( (time.perf_counter() - self.t_start) % 60 )
                print(f"surpass the tpm limits, wait for {t_rest} seconds...")
                time.sleep(t_rest)
                self.t_start = time.perf_counter()
            except openai.APITimeoutError as e:
                print("Timeout Error, Retrying...")
            except openai.APIConnectionError as e:
                print("Connect Error, Retrying...")
            
        outputs = completion.choices[0].message.content
        return outputs