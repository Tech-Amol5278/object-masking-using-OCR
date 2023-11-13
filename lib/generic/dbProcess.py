import pyodbc
import mysql.connector
from lib import load_configs as cf
from lib.generic import logger as lg


db_eng = cf.db_config_dict['db_engine'].lower()
def doConnect():

    # log_dir = configs_dict['log_dir']
    if db_eng == 'mssql':
        server, database, username, password = cf.db_config_dict['server'], cf.db_config_dict['database'], cf.db_config_dict['username'], cf.db_config_dict['password']
        conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        conn = pyodbc.connect(conn_str)
        lg.logthis(f"MSSQL >> Connection Established")
    elif db_eng == 'mysql':
        host, user, password, database = cf.db_config_dict['host'], cf.db_config_dict['user'], cf.db_config_dict['password'], cf.db_config_dict['database']
        conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
        lg.logthis(f"MYSQL >> Connection Established")
    elif db_eng == 'oracle':
        conn = mysql.connector.connect(host="localhost", user="yourusername", password="yourpassword", database="mydatabase")
        lg.logthis(f"ORACLE >> Connection Established")
    else :
        lg.logthis(f"Database not recognised")
    return conn
########################################
def doInsert(str):

    conn = doConnect()
    cursor = conn.cursor()   ## open db connection
    # insert_sql = f"INSERT INTO TB_INPUT_DETAILS (INPUT_FILE_NAME, INPUT_FILE_PATH, NEW_FILE_NAME, NEW_FILE_PATH) VALUES ('Manual_Name_1', 'Manual_Path_1', 'Manual_New_Name_1', 'Manual_New_Path_1' )"
    # insert_sql = str
    lg.logthis(str)
    cursor.execute(str)
    lg.logthis(f"db_engine >> {db_eng}")
    if db_eng == 'mssql':
        cursor.commit()
        cursor.close()   ## close db connection
    elif db_eng == 'mysql':
        conn.commit()
        pass
    elif db_eng == 'oracle':
        pass
    else:
        pass

    return
########## 03102023 ##########################
def doFetch(str):

    conn = doConnect()
    cursor = conn.cursor()   ## open db connection
    # insert_sql = f"INSERT INTO TB_INPUT_DETAILS (INPUT_FILE_NAME, INPUT_FILE_PATH, NEW_FILE_NAME, NEW_FILE_PATH) VALUES ('Manual_Name_1', 'Manual_Path_1', 'Manual_New_Name_1', 'Manual_New_Path_1' )"
    # insert_sql = str
    cursor.execute(str)
    db_result = cursor.fetchone()
    if db_eng == 'mssql':
        cursor.commit()
        cursor.close()   ## close db connection
    elif db_eng == 'mysql':
        conn.commit()
        pass
    elif db_eng == 'ORACLE':
        pass
    else:
         lg.logthis("Database not found")
    return db_result
######### 04102023 ##########################
def doUpdate(str):

    conn = doConnect()
    cursor = conn.cursor()   ## open db connection
    # insert_sql = f"INSERT INTO TB_INPUT_DETAILS (INPUT_FILE_NAME, INPUT_FILE_PATH, NEW_FILE_NAME, NEW_FILE_PATH) VALUES ('Manual_Name_1', 'Manual_Path_1', 'Manual_New_Name_1', 'Manual_New_Path_1' )"
    # insert_sql = str
    cursor.execute(str)
    if db_eng == 'mssql':
        cursor.commit()
        cursor.close()   ## close db connection
    elif db_eng == 'mysql':
        conn.commit()
        pass
    elif db_eng == 'oracle':
        pass
    else:
        lg.logthis("Database not found")
    return
######################################################################
def getMyDbFunct():
    print(f"db_engg >> {db_eng.lower()}")
    if db_eng.lower() == 'mssql':
        now = 'getdate()'
        pass
    elif db_eng.lower() == 'mysql':
        now = 'now()'
    elif db_eng.lower() == 'oracle':
        now = 'sysdate'
    else:
        lg.logthis("Database not found")

    return now
##############
now = getMyDbFunct()
#################################################
def getTimeTaken(interval, time1, time2):
    print(f"db_engg >> {db_eng.lower()}")
    if db_eng.lower() == 'mssql':
        timetaken = DATEDIFF(interval,time1,time2)
        pass
    elif db_eng.lower() == 'mysql':
        pass
    elif db_eng.lower() == 'oracle':
        pass
    else:
        lg.logthis("Database not found")

    return timetaken
#######
# timetaken = getTimeTaken(S, REQUEST_DATE,PROCESS_TIME)
