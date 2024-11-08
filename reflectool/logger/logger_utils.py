import os
import re

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def format_dict_str(dict_like, keys=None):
    pairs: list[str]
    if keys:
        pairs = [f"{k}: {dict_like[k]}" for k in keys]
    else:
        pairs = [f"{k}: {dict_like[k]}" for k in dict_like]
    return "\n".join(pairs)


def str_color_remove(color_str: str):
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    clean_str = ansi_escape.sub("", color_str)
    return clean_str

    
def check_log_file(log_file_name):
    if os.path.isdir(log_file_name):
        return True
    else:
        print(f"{log_file_name} does not exist. Created one")
        return False