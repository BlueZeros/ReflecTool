from reflectool.agents.TaskAgent import TaskAgent
from reflectool.agents.ModelAgent import ModelAgent
from reflectool.agents.CriticAgent import CriticAgent
from reflectool.agents.ReflexionAgent import ReflexionAgent
from reflectool.agents.ReflecToolAgent import ReflecToolAgent

def get_agent(agent_type):
    if agent_type == "agent":
        return TaskAgent
    
    elif agent_type == "model":
        return ModelAgent

    elif agent_type == "critic":
        return CriticAgent

    elif agent_type == "reflexion":
        return ReflexionAgent
    
    elif agent_type == "reflectool":
        return ReflecToolAgent
    
    else:
        raise NotImplementedError