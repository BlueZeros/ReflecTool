import os
import requests
import json
import difflib
import pandas as pd
from reflectool.actions.BaseAction import BaseAction, DEF_INNER_ACT_OBS
from reflectool.actions.actions_register import register


def max_similarity(synonyms, search_item):
    return max([difflib.SequenceMatcher(None, search_item.lower(), synonym.lower()).ratio() for synonym in synonyms])

# @register("Reflector")
class Reflection(BaseAction):
    def __init__(
        self,
        action_name="Reflection",
        action_desc="Use this action to reflect on the observation of previous actions, including the reliability of observations, the accuracy of action use, and the anticipation of the next action.",
        params_doc={"response": "this is the response of the reflection action"}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, response: str):
        return DEF_INNER_ACT_OBS
