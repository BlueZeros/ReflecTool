

DEF_INNER_ACT_OBS = "OK"
INNER_ACT_KEY = "response"

class AgentAction:
    def __init__(
        self,
        action_name="Agent_Action",
        action_desc="This is a instance of the agent action in the solving progress.",
        params: dict = {},
    ) -> None:
        self.action_name = action_name
        self.action_desc = action_desc
        self.params = params

class BaseAction:
    def __init__(
        self,
        action_name="Base_Action",
        action_desc="This is the base class for the action to complete the task",
        params_doc: dict = {},
        llm_drive: bool = False,
    ) -> None:
        """
        the agent action should be connected with data and env
        Input:
            action_name (str): action_name should be simple and distinctive.
                             One word string, concat with '_' or camel style.
            action_desc (str): agent use action_desc to understand this action
            params_doc (dict): a document to explain the input parameters to the API
        """
        
        self.action_name = action_name
        self.action_desc = action_desc
        self.params_doc = params_doc
        self.llm_drive = llm_drive

    def __call__(self, **kwargs) -> str:
        raise NotImplementedError

# @register("Think")
class Think(BaseAction):
    def __init__(
        self,
        action_name="Think",
        action_desc="Conduct thinking and reasoning process for solving task.",
        params_doc = {
            INNER_ACT_KEY: "this is your thinking response. Be specific and critical."
        }
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, **kwargs) -> str:
        return DEF_INNER_ACT_OBS

# @register("Plan")
class Plan(BaseAction):
    def __init__(self) -> None:
        action_name = "Plan"
        action_desc = "Plan step-by-step solutions for a task. Usually take at the beginning of the solving process."
        params_doc = {
            INNER_ACT_KEY: "this is the generated plans. Should decompose the task instructions as easy to execute steps."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, **kwargs):
        return DEF_INNER_ACT_OBS

# @register("Finish")
class Finish(BaseAction):
    def __init__(
        self,
        action_name="Finish",
        action_desc="Complete the task with a response.",
        params_doc = {
            INNER_ACT_KEY: "this is the finish action response. Respond with the final answer directly without other words."
        }
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, response):
        return str(response)
        


        
        
