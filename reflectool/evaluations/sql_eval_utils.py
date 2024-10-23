import os
import re
from reflectool.actions.EHRSQL import find_sqlite_files, post_process_sql, execute_sql, process_answer

def detect_sql(input_string):
    # 常见的SQL关键字列表
    sql_keywords = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 
        'TRUNCATE', 'MERGE', 'CALL', 'EXECUTE', 'DESC', 'SHOW', 'USE',
        'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
    ]
    
    # 创建一个正则表达式模式，匹配SQL关键字
    sql_pattern = re.compile(r'\b(?:' + '|'.join(sql_keywords) + r')\b', re.IGNORECASE)
    
    # 判断是否匹配SQL命令
    if re.search(sql_pattern, input_string):
        return True
    return False


def execute_sql_command(sql_query, data_base):
    sqlite_file = find_sqlite_files(data_base)
    # 
    # print(post_process_sql(sql_query))
    # print(sqlite_file)
    try:
        # import pdb
        # pdb.set_trace()
        sql = post_process_sql(sql_query, os.path.basename(os.path.normpath(data_base)))
        if sql is not None:
            result = execute_sql(sql, sqlite_file)
            result = process_answer(result)
            return result
        else:
            return None
    except TimeoutError as e:
        return None

    except:
        return None

