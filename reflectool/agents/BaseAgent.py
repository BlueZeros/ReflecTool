import os
import sys
from tqdm import tqdm

class BaseAgent:
    def __init__(self, args, name: str, role: str):
        # arguments
        for key, value in vars(args).items():
            setattr(self, key, value)

        # external arguments
        self.id = name
        self.name = name
        self.role = role
    
    def _build_prompt(self):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
    
    
    
