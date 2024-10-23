
import os
import json
from reflectool.actions.BaseAction import BaseAction
from reflectool.actions.actions_register import register

import requests
from bs4 import BeautifulSoup
from googlesearch import search

proxy = {
    "http": os.environ["bing_http"],
    "https": os.environ["bing_https"]
}

def extract_abstract(url, num_probe=10):
    try:
        response = requests.get(url, proxies=proxy)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # You can tweak this part based on the website structure
        paragraphs = soup.find_all('p')  # Find all paragraphs
        abstract = []
        
        # Extract only the first few paragraphs for abstract-like content
        for i in range(num_probe):  # Adjust the range based on how much you want
            abstract.append(paragraphs[i].text)
        
        return ' '.join(abstract)
    
    except Exception as e:
        return f"Error occurred: {e}"

@register("GoogleSearch", "Knowledge")
class GoogleSearch(BaseAction):
    def __init__(
        self,
        action_name="GoogleSearch",
        action_desc="Using this action to search online content with google.",
        params_doc={"query": "the search string. be simple."},
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, query):
        results = ""
        for url in search(query, num_results=2, proxy=proxy["http"]):
            abstract = extract_abstract(url)

            results += f"{abstract}\n\n"
        
        return results.strip()

# for url in search('uric acid', stop=2):
#     # print(url)
#     # informative_contents = extract_limited_webpage_text(url)
#     abstract = extract_abstract(url)
#     # print(url, informative_contents)
#     print(url, abstract)
#     print("*" * 100)