import os
import re
import json
import signal
import sqlite3
import pandas as pd
import multiprocessing
from ast import literal_eval
from reflectool.actions.actions_register import register
from reflectool.actions.BaseAction import BaseAction
from reflectool.models.base_model import Base_Model
from filelock import SoftFileLock

# This function loads and processes a database schema from a JSON file.

def load_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index > -1:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    db_schema = pd.DataFrame(schema, columns=['Database name', 'Table Name', 'Field Name', 'Type'])
    primary_key = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    foreign_key = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return db_schema, primary_key, foreign_key

# Generates a string representation of foreign key relationships in a MySQL-like format for a specific database.
def find_foreign_keys_MYSQL_like(foreign, db_id):
    df = foreign[foreign['Database name'] == db_id]
    output = "["
    for index, row in df.iterrows():
        output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ', '
    output = output[:-2] + "]"
    if len(output)==1:
        output = '[]'
    return output

# Creates a string representation of the fields (columns) in each table of a specific database, formatted in a MySQL-like syntax.
def find_fields_MYSQL_like(db_schema, db_id):
    df = db_schema[db_schema['Database name'] == db_id]
    df = df.groupby('Table Name')
    output = ""
    for name, group in df:
        output += "Table " +name+ ', columns = ['
        for index, row in group.iterrows():
            output += row["Field Name"]+', '
        output = output[:-2]
        output += "]\n"
    return output

# Generates a comprehensive textual prompt describing the database schema, including tables, columns, and foreign key relationships.
def create_schema_prompt(db_id, db_schema, primary_key, foreign_key, is_lower=True):
    prompt = find_fields_MYSQL_like(db_schema, db_id)
    prompt += "Foreign_keys = " + find_foreign_keys_MYSQL_like(foreign_key, db_id)
    if is_lower:
        prompt = prompt.lower()
    return prompt

__precomputed_dict = {
                    'temperature': (35.5, 38.1),
                    'sao2': (95.0, 100.0),
                    'heart rate': (60.0, 100.0),
                    'respiration': (12.0, 18.0),
                    'systolic bp': (90.0, 120.0),
                    'diastolic bp':(60.0, 90.0),
                    'mean bp': (60.0, 110.0)}
    
def post_process_sql(query, dataset="mimic_iv"):
    matches = list(re.finditer(r"```sql(.*?)```", query, re.DOTALL))
    if matches != []:
        for match in matches:
            query = match.group(1)
    else:
        # import pdb
        # pdb.set_trace()
        matches = list(re.finditer(r"```(.*?)```", query, re.DOTALL))
        if matches != []:
            for match in matches:
                query = match.group(1)

    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')

    if "current_time" in query:
        if dataset == "mimic_iv":
            __current_time = "2100-12-31 23:59:00"
        elif dataset == "mimic_iii" or dataset == "eicu":
            __current_time = "2105-12-31 23:59:00"
        else:
            raise NotImplementedError
        
        query = query.replace("current_time", f"'{__current_time}'")

    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list)==1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in __precomputed_dict:
                vital_range = __precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")

    query = query.replace("%y", "%Y").replace('%j', '%J')

    return query

def process_item(item):
    try:
        item = round(float(item),3)
    except:
        pass
    return str(item)

def process_answer(ans):
    try:
        ans = literal_eval(ans)
    except:
        pass
    if type(ans)==str:
        return ans
    else:
        return str(sorted([[process_item(c) for c in row] for row in ans])[:100]) # check only up to 100th record

# def execute_sql(sql, db_path, timeout=5):
#     pool = multiprocessing.Pool(processes=1)
#     result = pool.apply_async(execute_sql_task, (sql, db_path))
#     try:
#         return result.get(timeout=timeout)
#     except multiprocessing.TimeoutError:
#         # return None
#         raise TimeoutError(f"SQL query timed out after {timeout} seconds")
#     finally:
#         pool.terminate()

def execute_sql(sql, db_path):
    lock_path = f"{db_path}.lock"
    with SoftFileLock(lock_path):
        con = sqlite3.connect(db_path, uri=True)
        con.text_factory = lambda b: b.decode(errors="ignore")
        cur = con.cursor()
        result = cur.execute(sql).fetchall()

        con.close()
    return result

def find_sqlite_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sqlite"):
                return os.path.join(directory, file)

    return None

# DATA_PATH = "/mnt/petrelfs/liaoyusheng/projects/ClinicalAgent/ehrsql-2024/data"
# DB_ID = 'mimic_iv'
# TABLES_PATH = os.path.join(DATA_PATH, DB_ID, 'tables.json')   

def get_sql_mm_prompt(database_path):
    table_path = os.path.join(database_path, 'tables.json')   
    assert len(json.load(open(table_path, "r"))) == 1

    db_schema, primary_key, foreign_key = load_schema(table_path)
    db_id = json.load(open(table_path, "r"))[0]["db_id"]
    table_prompt = create_schema_prompt(db_id, db_schema, primary_key, foreign_key)
    system_msg = "Given the SQL tables structure above, your job is to transfer the user’s query into the sql command."
    return f"SQL Table:{table_prompt}\n\n{system_msg}"

def ehrsql_prompt(database_path):
    table_path = os.path.join(database_path, 'tables.json')   
    assert len(json.load(open(table_path, "r"))) == 1

    db_schema, primary_key, foreign_key = load_schema(table_path)
    db_id = json.load(open(table_path, "r"))[0]["db_id"]
    table_prompt = create_schema_prompt(db_id, db_schema, primary_key, foreign_key)
    system_msg = "Given the SQL tables information above, your job is to transfer the user’s query into the sql command in the code format and wrap the sql command within ```sql\n(.*?)\n```."
    
    dataset = os.path.basename(os.path.normpath(database_path))
    if dataset == "mimic_iv":
        __current_time = "2100-12-31 23:59:00"
    elif dataset == "mimic_iii" or dataset == "eicu":
        __current_time = "2105-12-31 23:59:00"
        
    return f"SQL Table:{table_prompt}\nThe current time is {__current_time}.\n\n{system_msg}"

# @register("LoadDB")
# class LoadDB(BaseAction):
#     def __init__(
#         self,
#         action_name="LoadDB",
#         action_desc="Use this action to load the table coloum information from SQL Data Base. ",
#         params_doc={"sql_database": "this is the path to the sql data base"}
#     ) -> None:
#         super().__init__(action_name, action_desc, params_doc)
    
#     def __call__(self, sql_database: str):
#         table_path = os.path.join(sql_database, 'tables.json')   
#         assert len(json.load(open(table_path, "r"))) == 1

#         db_schema, primary_key, foreign_key = load_schema(table_path)
#         db_id = json.load(open(table_path, "r"))[0]["db_id"]
#         table_prompt = create_schema_prompt(db_id, db_schema, primary_key, foreign_key)

#         return table_prompt

# @register("SQLExecutor")
# class SQLExecutor(BaseAction):
#     def __init__(
#         self,
#         action_name="SQLExecutor",
#         action_desc="Use this action to get information from SQL Data Base with SQL query.",
#         params_doc={"sql_database": "this is the path to the sql data base",
#                     "sql_query": "this is the sql query to get information from the sql_database"}
#     ) -> None:
#         super().__init__(action_name, action_desc, params_doc)
    
#     def __call__(self, sql_database: str, sql_query: str):
#         sqlite_file = find_sqlite_files(sql_database)

#         try:
#             sql_query = post_process_sql(sql_query, os.path.basename(os.path.normpath(sql_database)))
#             result = execute_sql(sql_query, sqlite_file)
#             result = process_answer(result)
#             return result
        
#         except Exception as e:
#             result = f'Error: {e}'
#             return f"Your previous SQL command: \n{sql_query}\n encountered an error with the feedback: {result}. Based on this, please rewrite this SQL command so that it correctly retrieves the results from the database."


DB_KNOWLEDGE = {
    "eicu": """Read the following data descriptions, generate the background knowledge as the context information that could be helpful for answering the question.
(1) Data include vital signs, laboratory measurements, medications, APACHE components, care plan information, admission diagnosis, patient history, time-stamped diagnoses from a structured problem list, and similarly chosen treatments.
(2) Data from each patient is collected into a common warehouse only if certain “interfaces” are available. Each interface is used to transform and load a certain type of data: vital sign interfaces incorporate vital signs, laboratory interfaces provide measurements on blood samples, and so on. 
(3) It is important to be aware that different care units may have different interfaces in place, and that the lack of an interface will result in no data being available for a given patient, even if those measurements were made in reality. The data is provided as a relational database, comprising multiple tables joined by keys.
(4) All the databases are used to record information associated to patient care, such as allergy, cost, diagnosis, intakeoutput, lab, medication, microlab, patient, treatment, vitalperiodic.
For different tables, they contain the following information:
(1) allergy: allergyid, patientunitstayid, drugname, allergyname, allergytime
(2) cost: costid, uniquepid, patienthealthsystemstayid, eventtype, eventid, chargetime, cost
(3) diagnosis: diagnosisid, patientunitstayid, icd9code, diagnosisname, diagnosistime
(4) intakeoutput: intakeoutputid, patientunitstayid, cellpath, celllabel, cellvaluenumeric, intakeoutputtime
(5) lab: labid, patientunitstayid, labname, labresult, labresulttime
(6) medication: medicationid, patientunitstayid, drugname, dosage, routeadmin, drugstarttime, drugstoptime
(7) microlab: microlabid, patientunitstayid, culturesite, organism, culturetakentime
(8) patient: patientunitstayid, patienthealthsystemstayid, gender, age, ethnicity, hospitalid, wardid, admissionheight, hospitaladmitsource, hospitaldischargestatus, admissionweight, dischargeweight, uniquepid, hospitaladmittime, unitadmittime, unitdischargetime, hospitaldischargetime
(9) treatment: treatmentid, patientunitstayid, treatmentname, treatmenttime
(10) vitalperiodic: vitalperiodicid, patientunitstayid, temperature, sao2, heartrate, respiration, systemicsystolic, systemicdiastolic, systemicmean, observationtime

Question: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205 on their current hospital encounter?
Knowledge:
- We can find the patient 035-2205 information in the patient database.
- As fluticasone-salmeterol 250-50 mcg/dose in aepb is a drug, we can find the drug information in the medication database.
- We can find the patientunitstayid in the patient database and use it to find the drug precsription information in the medication database.

Question: in the last hospital encounter, when was patient 031-22988's first microbiology test time?
Knowledge:
- We can find the patient 031-22988 information in the patient database.
- We can find the microbiology test information in the microlab database.
- We can find the patientunitstayid in the patient database and use it to find the microbiology test information in the microlab database.

Question: what is the minimum hospital cost for a drug with a name called albumin 5% since 6 years ago?
Knowledge:
- As albumin 5% is a drug, we can find the drug information in the medication database.
- We can find the patientunitstayid in the medication database and use it to find the patienthealthsystemstayid information in the patient database.
- We can use the patienthealthsystemstayid information to find the cost information in the cost database.

Question: what are the number of patients who have had a magnesium test the previous year?
Knowledge:
- As magnesium is a lab test, we can find the lab test information in the lab database.
- We can find the patientunitstayid in the lab database and use it to find the patient information in the patient database.

Question: {question}
Knowledge:""",

    "mimic_iii": """Read the following data descriptions, generate the background knowledge as the context information that could be helpful for answering the question.
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient, HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on ITEMID, it is possible to identify the concept represented by a given ITEMID.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
For different tables, they contain the following information:
(1) admissions: ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, MARITAL_STATUS, ETHNICITY, AGE
(2) chartevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(3) cost: ROW_ID, SUBJECT_ID, HADM_ID, EVENT_TYPE, EVENT_ID, CHARGETIME, COST
(4) d_icd_diagnoses: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(5) d_icd_procedures: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(6) d_items: ROW_ID, ITEMID, LABEL, LINKSTO
(7) d_labitems: ROW_ID, ITEMID, LABEL
(8) diagnoses_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(9) icustays: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, FIRST_CAREUNIT, LAST_CAREUNIT, FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME
(10) inputevents_cv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT
(11) labevents: ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(12) microbiologyevents: RROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME
(13) outputevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
(14) patients: ROW_ID, SUBJECT_ID, GENDER, DOB, DOD
(15) prescriptions: ROW_ID, SUBJECT_ID, HADM_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
(16) procedures_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(17) transfers: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, EVENTTYPE, CAREUNIT, WARDID, INTIME, OUTTIME

Question: What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?
Knowledge: 
- As comp-oth vasc dev/graft is a diagnose, the corresponding ICD9_CODE can be found in the d_icd_diagnoses database.
- The ICD9_CODE can be used to find the corresponding HADM_ID in the diagnoses_icd database.
- The HADM_ID can be used to find the corresponding COST in the cost database.

Question: had any tpn w/lipids been given to patient 2238 in their last hospital visit?
Knowledge: 
- We can find the visiting information of patient 2238 in the admissions database.
- As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
- As admissions only contains the visiting information of patients, we need to find the corresponding ICUSTAY_ID in the icustays database.
- We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit. 

Question: what was the name of the procedure that was given two or more times to patient 58730?
Knowledge: 
- We can find the visiting information of patient 58730 in the admissions database.
- As procedures are stored in the procedures_icd database, we can find the corresponding ICD9_CODE in the procedures_icd database.
- As we only need to find the name of the procedure, we can find the corresponding SHORT_TITLE as the name in the d_icd_procedures database.

Question: {question}
Knowledge:""",

    "mimic_iv": """Read the following data descriptions, generate the background knowledge as the context information that could be helpful for answering the question.
(1) Tables are linked by identifiers which usually have the suffix 'id'. For example, subject_id refers to a unique patient, hadm_id refers to a unique admission to the hospital.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single itemid which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on itemid, it is possible to identify the concept represented by a given itemid.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
For different tables, they contain the following information:
(1)admissions, columns: row_id, subject_id, hadm_id, admittime, dischtime, admission_type, admission_location, discharge_location, insurance, language, marital_status, age
(2)chartevents, columns: row_id, subject_id, hadm_id, stay_id, itemid, charttime, valuenum, valueuom
(3)cost, columns: row_id, subject_id, hadm_id, event_type, event_id, chargetime, cost
(4)d_icd_diagnoses, columns: row_id, icd_code, long_title
(5)d_icd_procedures, columns: row_id, icd_code, long_title
(6)d_items, columns: row_id, itemid, label, abbreviation, linksto
(7)d_labitems, columns: row_id, itemid, label
(8)diagnoses_icd, columns: row_id, subject_id, hadm_id, icd_code, charttime
(9)icustays, columns: row_id, subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime
(10)inputevents, columns: row_id, subject_id, hadm_id, stay_id, starttime, itemid, amount
(11)labevents, columns: row_id, subject_id, hadm_id, itemid, charttime, valuenum, valueuom
(12)microbiologyevents, columns: row_id, subject_id, hadm_id, charttime, spec_type_desc, test_name, org_name
(13)outputevents, columns: row_id, subject_id, hadm_id, stay_id, charttime, itemid, value
(14)patients, columns: row_id, subject_id, gender, dob, dod
(15)prescriptions, columns: row_id, subject_id, hadm_id, starttime, stoptime, drug, dose_val_rx, dose_unit_rx, route
(16)procedures_icd, columns: row_id, subject_id, hadm_id, icd_code, charttime
(17)transfers, columns: row_id, subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime

Question: What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?
Knowledge: 
- As comp-oth vasc dev/graft is a diagnose, the corresponding icd_code can be found in the d_icd_diagnoses database.
- The icd_code can be used to find the corresponding hadm_id in the diagnoses_icd database.
- The hadm_id can be used to find the corresponding cost in the cost database.

Question: had any tpn w/lipids been given to patient 2238 in their last hospital visit?
Knowledge: 
- We can find the visiting information of patient 2238 in the admissions database.
- As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
- We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit. 

Question: what was the name of the procedure that was given two or more times to patient 58730?
Knowledge: 
- We can find the visiting information of patient 58730 in the admissions database.
- As procedures are stored in the procedures_icd database, we can find the corresponding icd_code in the procedures_icd database.
- As we only need to find the name of the procedure, we can find the corresponding long_title as the name in the d_icd_procedures database.

Question: {question}
Knowledge:"""
}

@register("DBManual", "Numerical")
class DBManual(BaseAction):
    def __init__(
        self,
        action_name="DBManual",
        action_desc="Use this action to obtain the sql database description and usage method related to the query. This action is helpful when SQLCoder cannot find the information.",
        params_doc={"sql_database": "this is the path to the sql data base",
                    "query": "this is the query in natural language. Usually it is the instruction of the task."},
        llm_drive=True
    ) -> None:
        super().__init__(action_name, action_desc, params_doc, llm_drive)

    def __call__(self, sql_database: str, query: str, llm: Base_Model) -> str:
        database_name = os.path.basename(os.path.normpath(sql_database))
        if database_name in DB_KNOWLEDGE:
            prompt = DB_KNOWLEDGE[database_name]
            prompt = prompt.format(question=query)
            return llm(prompt).split("Question:")[0].strip()
        else:
            return f"THe Manual does not contain the information about the SQL database: {sql_database}."

# @register("Calendar")
# class Calendar(BaseAction):
#     def __init__(
#         self,
#         action_name="Calendar",
#         action_desc="Use this action to get the specific date and time in the sql database.",
#         params_doc={"sql_database": "this is the path to the sql data base",
#                     "time_gap": "This is the difference between the time you want to know and the current time. You can get the Timestamp of the last year at this time with Calendar[{\"time_gap\": \"-1 year\"}]."},
#     ) -> None:
#         super().__init__(action_name, action_desc, params_doc)
    
#     def __call__(self, sql_database: str, time_gap: str) -> str:
#         sqlite_file = find_sqlite_files(sql_database)
#         try:
#             sql_command = "select datetime(current_time, '{}')".format(time_gap)
#             result = execute_sql(post_process_sql(sql_command, os.path.basename(os.path.normpath(sql_database))), sqlite_file)
#             result = process_answer(result)
#         except:
#             raise Exception("The date calculator {} is incorrect. Please check the syntax and make necessary changes. For the current date and time, please call Calendar('0 year').".format(time_gap))
        
#         return result

@register("SQLCoder", "Numerical")
class SQLCoder(BaseAction):
    def __init__(
        self,
        action_name="SQLCoder",
        action_desc="Use this action to gather the patient information from the sql_database. The SQLCoder will transfer the natural language query into the SQL command and get the information from the sql_database.",
        params_doc={"sql_database": "this is the path to the sql data base",
                    "query": "this is the query to search information in natural language. More precise query such as getting information from a column in a table, can increase the chances of successfuly gathering information."},
        llm_drive=True,
    ) -> None:
        super().__init__(action_name, action_desc, params_doc, llm_drive)


    def __call__(self, sql_database: str, query: str, llm: Base_Model, skip_indicator="null") -> str:
        prompt = ehrsql_prompt(sql_database)
        sqlite_file = find_sqlite_files(sql_database)
        input_query = f"{prompt}\nUser's query: {query}\nSQL command:"

        n = 0
        while n < 5:
            n += 1
            sql_query = llm(input_query)

            if sql_query != skip_indicator:
                try:
                    sql_query = post_process_sql(sql_query, os.path.basename(os.path.normpath(sql_database)))
                    result = execute_sql(sql_query, sqlite_file)
                    result = process_answer(result)
                    return result
                except Exception as e:
                    result = f'Error: {e}'
                    print(f"[Error]: {result}")
                    input_query = f"{prompt}\nUser's query: {query}\n\nYour previous SQL command: \n{sql_query}\n encountered an error with the feedback: {result}. Based on this, please rewrite this SQL command so that it correctly retrieves the results from the database."

            else:
                return skip_indicator
        
        return f"Unable to successfully convert the user query into SQL command with {result}."




