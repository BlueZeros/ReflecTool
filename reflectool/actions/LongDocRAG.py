import os
import json
from reflectool.actions.actions_register import register
from reflectool.actions.BaseAction import BaseAction
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage

def load_multiple_documents(folder_path):
    combined_content = ""
    
    for filename in os.listdir(folder_path):
        # 只处理 .txt 文件
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # 拼接成指定格式的字符串
                combined_content += f"[{filename}]:\n{content}\n"
    
    return combined_content

@register("LongDocRAG", "Data")
class LongDocRAG(BaseAction):
    def __init__(
        self,
        action_name="LongDocRAG",
        action_desc="Using this action to construct a retrival knowledge base from the upload files and query the information from the knowledge base. The action can only be token when the upload files is not None",
        params_doc={"path": "The path of the upload files, can either be a folder or a file", "query": "the query to search information"},
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, path, query):
        
        if os.path.exists(f"{path}/persist"):
            storage_context = StorageContext.from_defaults(persist_dir=f"{path}/persist")
            index = load_index_from_storage(storage_context)
        else:
            documents = SimpleDirectoryReader(path).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=f"{path}/persist")
        
        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        return str(response)