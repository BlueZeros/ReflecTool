import os
import sys
import json
from tqdm import tqdm
from reflectool.utilities import *
from reflectool.logger.logger import AgentLogger
from reflectool.agents.TrainAgent import TrainAgent
from reflectool.datas.DataManager import DataManager
from reflectool.train.action_reflector_optimizer import ActionReflectorOptimizer

class AgentTrainer():

    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)

        logger = AgentLogger(
            FLAG_PRINT=args.log_print,
            PROMPT_DEBUG_FLAG=args.prompt_debug,
            log_file_name=args.log_file_name
        )

        self.agent = TrainAgent(
            args, 
            logger=logger
        )
    
        self.data_manager = DataManager(args)
        self.optimizer = ActionReflectorOptimizer(self.agent, output_path=self.output_path)
        self.batch_size = 10

        if self.resume:
            self.resume_optimizer()
    
    def train(self):
        batch_task_log = []
        for sample_num, task in tqdm(enumerate(self.data_manager), ncols=60):
            task_log = self.agent(task)
            self.data_manager.add_cache(task_log)

            if task_log["score"] > 0.7:
                batch_task_log.append(task_log)

        # for sample_num, task_log in tqdm(enumerate(self.data_manager.cache), ncols=60):
        #     batch_task_log.append(task_log)

            if batch_task_log != [] and len(batch_task_log) % self.batch_size == 0:
                self.optimizer.calculate_loss(batch_task_log)
                self.optimizer.backward()
                self.optimizer.step()

                batch_task_log = []
                self.save()
    
    def resume_optimizer(self):
        self.optimizer.resume()
    
    def save(self):
        self.data_manager.save_cache()
        self.data_manager.score()
        self.optimizer.save()
    
        
            



    


    

