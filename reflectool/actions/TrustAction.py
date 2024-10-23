import os
import json
import requests
from googleapiclient import discovery
from reflectool.actions.actions_register import register
from reflectool.actions.BaseAction import BaseAction
from privateai_client import PAIClient
from privateai_client import request_objects


# @register("ToxicityDetector")
class ToxicityDetector(BaseAction):
    def __init__(
        self,
        action_name="ToxicityDetector",
        action_desc="Using this action to detect the toxicity score of the response.",
        params_doc={"sentence": "Sentences to be detected for toxicity score."},
    ) -> None:
        
        super().__init__(action_name, action_desc, params_doc)

        API_KEY = os.environ["PERSPECTIVE_KEY"] 

        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
    
    def __call__(self, sentence):
        analyze_request = {
            'comment': { 'text': sentence },
            'requestedAttributes': {'TOXICITY': {}}
            }

        response = self.client.comments().analyze(body=analyze_request).execute()
        score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

        return f"The toxicity probability of the input sentence is {score}"


# @register("PrivacyDetector")
class PrivacyDetector(BaseAction):
    def __init__(
        self,
        action_name="PrivacyDetector",
        action_desc="Using this action to prevent the privacy identity information leakage in the response.",
        params_doc={"response": "The response to be preprocessed"},
    ) -> None:
        
        super().__init__(action_name, action_desc, params_doc)

        self.client = PAIClient(url="https://api.private-ai.com/community", api_key=os.environ["PRIVATEAI_KEY"] )

    def __call__(self, response: str):

        text_request = request_objects.process_text_obj(text=[response], processed_text={"type": "SYNTHETIC"})
        result = self.client.process_text(text_request)

        return result.processed_text[0]

