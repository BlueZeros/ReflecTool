import os
import sys
import json
import argparse
import random
from tqdm import tqdm
from reflectool.utilities import parse_args
from reflectool.train.trainer import AgentTrainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    trainer = AgentTrainer(args)
    trainer.train()

    
