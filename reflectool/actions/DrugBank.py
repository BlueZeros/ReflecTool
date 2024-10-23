import os
import requests
import json
import difflib
import pandas as pd
from functools import lru_cache 

from reflectool.actions.BaseAction import BaseAction
from reflectool.actions.actions_register import register


def max_similarity(synonyms, search_item):
    return max([difflib.SequenceMatcher(None, search_item.lower(), synonym.lower()).ratio() for synonym in synonyms])

@register("DrugBank", "Knowledge")
class DrugBank(BaseAction):
    def __init__(
        self,
        action_name="DrugBank",
        action_desc="Use this action to search the information about specific drug",
        params_doc={"drug_name": "this is the name of the drug to be searched"}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)

        self.df = pd.read_pickle('./drugbank/drugbank.pkl')
    
    @lru_cache(maxsize=256)
    def __call__(self, drug_name: str):
        
        self.df['similarity'] = self.df['synonyms'].apply(lambda x: max_similarity(x, drug_name))
        most_similar_row = self.df.loc[self.df['similarity'].idxmax()]
        self.df = self.df.drop(columns=['similarity'])

        most_similar_dict = most_similar_row.to_dict()  # 转换为字典
        del most_similar_row["similarity"]
        result = json.dumps(most_similar_dict)  # 转换为JSON格式字符串
        return result

