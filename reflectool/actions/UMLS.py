import os
import re
import json
import torch
import requests
import transformers
from functools import lru_cache 

from reflectool.actions.BaseAction import BaseAction
from reflectool.actions.actions_register import register

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

        self.proxy = {
            "http": os.environ["bing_http"],
            "https": os.environ["bing_https"]
        }

    def search_cui(self, query):
        cui_results=[]

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query, proxies=self.proxy)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found."+"\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_definitions(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix, proxies=self.proxy)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs  = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_relations(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "relations", self.apikey)
            r = requests.get(self.content_url + suffix, proxies=self.proxy)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs  = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)



@register("UMLS", "Knowledge")
class UMLS(BaseAction):
    def __init__(
        self,
        action_name="UMLS",
        action_desc="Use this action to query the definition and the related medical concept of the medical_terminology.",
        params_doc={"medical_terminology": "Biomedical terminology, should be a phrase or a word"}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)

        self.umls_api = UMLS_API(os.getenv("UMLS_API_KEY"))
    
    @lru_cache(maxsize=256)
    def __call__(self, medical_terminology: str):
        umls_res = {}

        # for key in medical_terminologies:
        cuis = self.umls_api.search_cui(medical_terminology)

        if len(cuis) == 0:
            return "No relevant information!"

        cui = cuis[0][0]
        name = cuis[0][1]

        defi = ""
        definitions = self.umls_api.get_definitions(cui)

        if definitions is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None

            for definition in definitions:
                source = definition["rootSource"]
                if source == "MSH":
                    msh_def = definition["value"]
                    break
                elif source == "NCI":
                    nci_def = definition["value"]
                elif source == "ICF":
                    icf_def = definition["value"]
                elif source == "CSP":
                    csp_def = definition["value"]
                elif source == "HPO":
                    hpo_def = definition["value"]

            defi = msh_def or nci_def or icf_def or csp_def or hpo_def

        relations = self.umls_api.get_relations(cui)
        rels = []

        if relations is not None:
            for rel in relations[:]:
                related_from_id_name = rel.get("relatedFromIdName", "")
                additional_relation_label = rel.get("additionalRelationLabel", "")
                related_id_name = rel.get("relatedIdName", "")

                if related_from_id_name:
                    rels.append((related_from_id_name, additional_relation_label, related_id_name))

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels[:5]}

        context = ""
        for k, v in umls_res.items():
            name = v["name"]
            definition = v["definition"]
            rels = v["rels"]
            rels_text = ""
            for rel in rels:
                rels_text += "(" + rel[0] + "," + rel[1] + "," + rel[2] + ")\n"
            text = f"Name: {name}\nDefinition: {definition}\n"
            if rels_text != "":
                text += f"Relations: {rels_text}"

            context += text + "\n"
            if context != "":
                context = context[:-1]

        return context