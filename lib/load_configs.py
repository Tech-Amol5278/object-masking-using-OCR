import platform
import os
from lib import functions as fnc
from lib.generic import logger as lg

#############################################
# if 'windows' in platform.platform().lower():
#     print(f"Operating on {platform.platform()}")
#     # lg.logthis(f"Operating on {platform.platform()}")
#     config_path = r'D:\Data_Science\ds\challenges\OCR\kyc_masking\kyc_masking_test\config\sys_config.properties'
# elif 'linux' in platform.platform().lower():
#     print(f"Operating on {platform.platform()}")
#     # lg.logthis(f"Operating on {platform.platform()}")
#     config_path = r'/home/server/ocr/kyc_masking/config/linx_ocr.properties'
# else:
#     pass
################################################
# parent_dir = r"D:\Data_Science\ds\challenges\OCR\kyc_masking\kyc_masking_test"
curr_dir = os.getcwd()
parent_dir = curr_dir
configs_dir = os.path.join(parent_dir,'config')
sys_config = os.path.join(configs_dir,'sys_config.properties')
ocr_config = os.path.join(configs_dir,'ocr_config.properties')
db_config_path = os.path.join(configs_dir,'dbconfig.properties')

print('curr_dir', curr_dir)
print('parent_dir', parent_dir)
print('configs_dir', configs_dir)
print('sys_config', sys_config)
print('ocr_config', ocr_config)
print('db_config_path', db_config_path)
########################################################################
sys_conf_dict = fnc.getConfig(sys_config)
ocr_conf_dict = fnc.getConfig(ocr_config)
#################### Auto assignment system configs #########################
for key, val in sys_conf_dict.items():
    locals()[key] = val
    print(f"{key} = {val}")
#################### Auto assignment ocr configs #########################
for key, val in ocr_conf_dict.items():
    locals()[key] = val
    print(f"{key} = {val}")
######### Load Db Configs ###############################################
db_config_dict = fnc.getConfig(db_config_path)
db_engine = db_config_dict['db_engine']
###################### Test ###############################################





