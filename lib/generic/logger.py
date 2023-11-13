from lib import load_configs as cf
import datetime as dt
import os

def logthis_old(text):
    '''
    :param log_dir: logs directory
    :param text: text to be logged
    :return:

    [INFO] [18 Feb 2020 10:59:16,400] [biz.infrasofttech.aml.controller.ProcessController] : Starting initialisation
    '''

    from datetime import datetime
    curr_date = datetime.now()
    curr_date = str(curr_date.strftime("%d%m%Y"))
    # from datetime import datetime
    # current_time = str(datetime.now().strftime("%d%m%Y"))
    file = curr_date + '_ocr_log.log'
    log_file = os.path.join(cf.log_dir,file)

    f = open(log_file, "a")
    f.write(str("\n"+ "[INFO]" + "[" + datetime.now().strftime("%d %b %Y %H:%M:%S"))  + "] " + text)
    f.close()
    return


def logthis(text):
    '''
    :param log_dir: logs directory
    :param text: text to be logged
    :return:

    [INFO] [18 Feb 2020 10:59:16,400] [biz.infrasofttech.aml.controller.ProcessController] : Starting initialisation
    '''
    log_dir = cf.log_dir
    log_name = cf.log_name
    log_ext = cf.log_ext
    log_file = log_dir + log_name + log_ext

    if os.path.exists(log_file):
        logm_time = os.path.getmtime(log_file) # modification time
        # print(old_time)
        logm_date = dt.date.fromtimestamp(logm_time)
        # print(old_date)
        curr_date = dt.date.today()
        # print(curr_date)
        # to_rename = (curr_date - datetime.timedelta(days=1)).strftime("%d%m%Y")
        # to_rename = file + "_" + to_rename
        # rename_log_file = os.path.join(log_dir,file)

        '''
        Logic : If log file exists rename it to last_modified_date and create new log file
        '''

        if logm_date == curr_date:
            write_file = log_file
        else :
            ## Rename existing file
            logm = logm_date.strftime("%d%m%Y")
            logm_f_ = log_name + "_" + logm + log_ext
            logm_file = os.path.join(log_dir,logm_f_)
            os.rename(log_file, logm_file)
            ## Create new file
            write_file = log_file

    else:
        write_file = log_dir + log_name + log_ext
    ############### write to a log file ###############################
    f = open(write_file, "a")
    f.write(str("\n"+ "[INFO]" + "[" + dt.datetime.now().strftime("%d %b %Y %H:%M:%S"))  + "] " + text)
    f.close()
    return


