import os
import copy
import json
import numpy as np
from typing import Dict, Union

from reflectool.commons.TaskPackage import TaskPackage
from reflectool.actions.BaseAction import AgentAction
from reflectool.evaluations.eval_score import score_task
from reflectool.memory.memory_utils import load_memory_as_dict, load_memory_list_format, format_memory
from reflectool.memory.memory_utils import MEMORY_TASK_KEY, MEMORY_ACT_OBS_KEY, MEMORY_PREV_ACT_OBS_KEY
from reflectool.agent_prompts.prompt_utils import task_chain_format

from pyserini.search.lucene import LuceneSearcher

class AgentMemory:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.memory = None

    def get_action_chain(self, task: TaskPackage):
        raise NotImplementedError

    def add_action(self, action: AgentAction):
        raise NotImplementedError

    def add_new_task(self, task: TaskPackage):
        raise NotImplementedError

    def add_act_obs(self, task: TaskPackage, action: AgentAction, observation: str):
        raise NotImplementedError


class ShortTermMemory(AgentMemory):
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.memory: Dict[str, Dict[str, Union[TaskPackage, list]]] = {}

    def add_new_task(self, task: TaskPackage):
        self.memory[task.task_id] = {MEMORY_TASK_KEY: task, MEMORY_ACT_OBS_KEY: [], MEMORY_PREV_ACT_OBS_KEY: []}

    def get_action_chain(self, task: TaskPackage):
        return self.memory[task.task_id][MEMORY_ACT_OBS_KEY]

    def get_prev_action_chain(self, task: TaskPackage):
        return self.memory[task.task_id][MEMORY_PREV_ACT_OBS_KEY]
    
    def delete_task(self, task: TaskPackage):
        del self.memory[task.task_id]

    def add_act_obs(self, task: TaskPackage, action: AgentAction, observation: str = ""):
        """adding action and its corresponding observations into memory"""
        self.memory[task.task_id][MEMORY_ACT_OBS_KEY].append((action, observation))
    
    def add_prev_act_obs(self, task: TaskPackage, act_obs: list):
        """adding previous action and its corresponding observations into memory"""
        self.memory[task.task_id][MEMORY_PREV_ACT_OBS_KEY].append(act_obs)
    
    def update(self, task: TaskPackage):
        """transfer the short term memory into the previous memory"""
        self.memory[task.task_id][MEMORY_PREV_ACT_OBS_KEY].append(self.memory[task.task_id][MEMORY_ACT_OBS_KEY])
        self.memory[task.task_id][MEMORY_ACT_OBS_KEY] = []


class LongTermMemory(AgentMemory):
    def __init__(self, agent_id: str, memory_path: str = "./memory/memory_bank") -> None:
        self.agent_id = agent_id
        # self.memory: Dict[str, Dict[str, Union[TaskPackage, list]]] = {}
        self.memory_path = memory_path
        self.__init_memory__()
    
    def __init_memory__(self):
        self.memory = load_memory_list_format(self.memory_path)

    def __get_memory__(self, indice: int):
        return self.memory[indice]

    def get_memories(self, indices: list[int] = None):
        memory_len = len(self.memory)
        if memory_len == 0:
            return None
        
        if indices is None:
            indices = list(range(memory_len))
            examples = [self.__get_memory__(idx) for idx in indices]
        return examples

TASK_MEMORY_MAPPING = {
    "medqa": "medqa",
    "mmlu": "medqa",
    "bioasq": "medqa",
    "pubmedqa": "medqa",
    "medcalc": "medcalc",
    "ehrsql": "ehrsql",
    "mimic_iii": "ehrsql",
    "eicu": "ehrsql",
    "medmentions": "medmentions",
    "emrqa": "emrqa",
    "longhealthqa": "",
    "medhalt_rht": "medqa",
    "ehr_halt": "ehrsql",
}

class WritenLongTermMemory(AgentMemory):
    def __init__(self, agent_id, exp_name, memory_path="../data/memory/", task="test", split="medqa", k=1, memory_type="standard", write_mode=False, update_freq=10, memory_wo_reflect=False) -> None:
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.k = k
        self.memory_type = memory_type
        self.split = split
        self.memory_wo_reflect = memory_wo_reflect

        if self.memory_type in ["standard", "critic_standard", "reflexion_standard"]:
            self.memory_path = os.path.join(memory_path, memory_type)
        
        elif self.memory_type in ["task_standard"]:
            self.memory_path = os.path.join(memory_path, memory_type, TASK_MEMORY_MAPPING[split])

        else:
            if write_mode:
                self.memory_path = os.path.join(memory_path, memory_type, task, split, exp_name)
            else:
                self.memory_path = memory_path

        self.write_mode = write_mode
        self.update_freq = update_freq
        self.memory_buffer = []

        self.__load_memory__()

    def __load_memory__(self):
        self.index_dir = os.path.join(self.memory_path, "index")

        if self.write_mode:
            if not os.path.exists(self.memory_path):
                os.makedirs(self.memory_path, exist_ok=True)

            if not os.path.exists(self.index_dir):
                os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(self.memory_path, self.index_dir))
            self.index = LuceneSearcher(self.index_dir)

        else:
            assert os.path.exists(self.index_dir) or self.k == 0
            if self.k != 0 and "standard" not in self.memory_type:
                self.index = LuceneSearcher(self.index_dir)
    
    def __save_memory__(self, task):
        task_score = task["score"] if "score" in task else score_task(self.split, task)
        if self.write_mode and task_score > 0.7:
            memory_example = copy.deepcopy(task)
            memory_example["contents"] = task_chain_format(TaskPackage(**task["task"]), [])

            self.memory_buffer.append(memory_example)

            if len(self.memory_buffer) >= self.update_freq:
                self.__write_memory__()

    def __write_memory__(self):
        for id, memory_example in enumerate(self.memory_buffer):
            write_path = os.path.join(self.memory_path, f'example_{self.index.num_docs + id + 1}.json')
            memory_example["id"] = f"example_{self.index.num_docs + id + 1}"
            with open(write_path, "w") as f:
                json.dump(memory_example, f, indent=4, ensure_ascii=False)
        
        self.__load_memory__()
        self.memory_buffer = []
    
    def get_memory(self, task):
        if "standard" in self.memory_type:
            memories = []
            for i in range(self.k):
                if os.path.exists(os.path.join(self.memory_path, f'example_{i + 1}.json')):
                    memories.append(format_memory(json.load(open(os.path.join(self.memory_path, f'example_{i + 1}.json'))), self.memory_wo_reflect))
                else:
                    raise FileNotFoundError(f"Error: The file '{os.path.join(self.memory_path, f'example_{i + 1}.json')}' was not found. Please check the file path and try again.")

        else:
            query = task_chain_format(task, [])
            memories = self.index.search(query, k=self.k)
            # memories_score = np.array([m.score for m in memories])
            ids = [m.docid for m in memories]
            memories = [format_memory(json.load(open(os.path.join(self.memory_path, f"{id}.json"))), self.memory_wo_reflect) for id in ids]

        return memories

    
    def get_memories(self, task):
        if self.k == 0:
            return []
        
        memories = self.get_memory(task)
        return memories
    
    def __len__(self):
        return self.index.num_docs




        

