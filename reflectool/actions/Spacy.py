import os
import json
import spacy
from googleapiclient import discovery
from reflectool.actions.actions_register import register
from reflectool.actions.BaseAction import BaseAction


@register("EntityRecognizor", "Data")
class EntityRecognizor(BaseAction):
    def __init__(
        self,
        action_name="EntityRecognizor",
        action_desc="Using this action to recognize the biomedical entities in the sentence.",
        params_doc={"sentence": "Sentences to be recognized the entities."},
    ) -> None:
        
        super().__init__(action_name, action_desc, params_doc)

        self.recog = spacy.load("en_core_sci_sm")
        # self.recog.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    def __call__(self, sentence):
        doc = self.recog(sentence)
        entities = doc.ents

        return f"Biomedical Entity List: {entities}"