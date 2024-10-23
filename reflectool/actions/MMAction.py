from PIL import Image
from reflectool.models import get_model
from reflectool.utilities import extract_python_code, execute_python_code
from reflectool.actions.BaseAction import BaseAction
from reflectool.actions.actions_register import register


@register("MiniCPM", "MultiModal")
class MiniCPM(BaseAction):
    def __init__(
        self,
        action_name="MiniCPM",
        action_desc="Use this action to gather information from the medical image with a general multi-modal large langugage model.",
        params_doc={"query": "this is the query string", "image": "this requires a image input"}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)

        self.llm = get_model("minicpm-v-2.6", cuda_id=1)
    
    def __call__(self, query, image) -> str:
        output = self.llm(query, image) # generates
        # print(generated_text)
        return output


@register("HuatuoGPT", "MultiModal")
class HuatuoGPT(BaseAction):
    def __init__(
        self,
        action_name="HuatuoGPT",
        action_desc="Use this action to gather information from the medical image with a medical-domain multi-modal large language model.",
        params_doc={"query": "this is the query string", "image": "this requires a image input"}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.llm = Blip2ForConditionalGeneration.from_pretrained("/mnt/hwfile/medai/liaoyusheng/checkpoints/blip2-flan-t5-xl", torch_dtype=torch.float16).to(device)
        # self.llm = HuatuoChatbot("/mnt/hwfile/medai/LLMModels/Model/HuatuoGPT-Vision-7B", device=device) # loads the model 
        self.llm = get_model("huatuo-vision-7b", cuda_id=1)

    def __call__(self, query, image) -> str:
        output = self.llm(query, image) # generates
        # print(generated_text)
        return output

@register("MedCaptioner", "MultiModal")
class MedCaptioner(BaseAction):
    def __init__(
        self,
        action_name="MedCaptioner",
        action_desc="Use this action to generate comprehensive caption for the medical image with a medical captioner.",
        params_doc={"image": "this requires a image input"}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)

        self.llm = get_model("llavamedpp", cuda_id=1)
    
    def __call__(self, image):
        output = self.llm(image)
        return output

