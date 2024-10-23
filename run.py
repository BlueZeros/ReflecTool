import os
import sys
import random
from tqdm import tqdm
from reflectool.utilities import parse_args
from reflectool.agents import get_agent
from reflectool.logger.logger import AgentLogger
from reflectool.datas.DataManager import DataManager

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

if __name__ == "__main__":

    args = parse_args()
    random.seed(args.seed)

    # Build logger
    agent_logger = AgentLogger(
        FLAG_PRINT=args.log_print,
        PROMPT_DEBUG_FLAG=args.prompt_debug,
        log_file_name=args.log_file_name
    )

    agent_class = get_agent(args.agent)
    # Build the solver
    agent = agent_class(
        args, 
        logger=agent_logger)

    # Build DataLoader
    data_manager = DataManager(args)
    
    for task_id, task in tqdm(enumerate(data_manager), ncols=60):
        task_log = agent(task)
        data_manager.add_cache(task_log)
        data_manager.save_task(task_log)
        data_manager.score()
