import pandas as pd
from lib.generic import logger as lg
import platform
###################################################################################
def getConfig (file_path):
    print('Function Start : getConfig')
    from pathlib import Path
    '''
    Function to get the configuration from properties file
    Logic :
        Requires two arguements to be passed in below parameters respectively
            file_path : full path of the file including ex. "C:\abc\filename.txt"
            search_text : a string which is to be searched within the given file_path

        This returns the text after the ':' sign on the line where the above search_text is found.
    '''

    path_ls = ['tesseract_executable_path', 'ocr_parent_dir', 'upload_dir', 'output_dir', 'work_dir']

    with open(file_path, 'r') as file:
        # Read all lines
        lines = file.readlines()
        configs_dict = {}
        # iterate over the lines
        for line in lines:
            if line.startswith("#") == False:
                if line.startswith("'") == False:
                    if ":" in line:
                        ls = line.split(":",maxsplit=1)
                        configs_dict.update({ls[0].strip() : ls[1].strip()})
    #
    return configs_dict
###################################################################################
def ocr_aadhar_base(ocr_dir, filename):
    import cv2
    import pytesseract
    # import pandas as pd
    from lib import loadData as ld, textProcessing as tp, functions as fnc, imageProcessing as imp
    # from PIL import Image as im

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # load data
    img, img_file, image_dir, work_dir, config_file = ld.loadData(ocr_dir, filename)
    # # load Tesseract
    # config = fnc.getConfig(config_file, 'tesseract_executable_path')
    # # pytesseract.pytesseract.tesseract_cmd = r"{}".format(config_file)
    # pytesseract.pytesseract.tesseract_cmd = str(config)
    # # Set an environment variable
    # # import os
    # # os.environ['PATH'] = str(config)
    print('Function Start : ocr_wrapper_aadhar')

    # Binarization
    thrshldedImg = imp.get_binary_image(img_file, 'adapt_thres')
    # Save image
    savedThrshldedImg = work_dir + '\\_thresholdedImg.png'

    cv2.imwrite(savedThrshldedImg, thrshldedImg)
    skewCorrectedImg = imp.doSkewCorrection(savedThrshldedImg)

    # Save image
    savedSkewCorrectedImg = work_dir + '\\_skewCorrectedImg.png'
    skewCorrectedImg.save(savedSkewCorrectedImg)

    # convert to text
    text = pytesseract.image_to_string(img)

    # write the extracted string to text file
    noext = filename.split('.')[0]
    extr_file = work_dir+'\\'+noext+'.ext'
    with open(extr_file, "w") as file:
        file.write(text)

    # organize information
    uid_dict = tp.doTextProcess(text)

    # test
    # print(uid_dict)
    print('filename : ', filename)
    print('noext : ', noext)
    print('work_dir : ', work_dir)
    print('extr_file : ', extr_file)

    # Clean file
    clean_file = tp.doClean(extr_file, noext, work_dir)

    # get Name
    uid_name, name_line_index = tp.getStrForward(clean_file,'Enrollment',3)
    print('uid_name : ', uid_name)

    # get YOB
    uid_yob, yob_line_index = tp.getYob(clean_file,'Year of Birth',0)
    uid_yob = uid_yob.split(':')[1]
    print('uid_yob: ', uid_yob)

    # get UID
    uid, uid_line_index = tp.getStrForward(clean_file,'Ref:',3)
    print('uid : ', uid)

    # uid_dict
    # uid_dict = {}
    # uid_dict['filename'] = filename
    # uid_dict['uid_name'] = uid_name
    # uid_dict['uid_yob'] = uid_yob
    # uid_dict['uid'] = uid
    #
    # print(uid_dict)


    return filename, uid_name, uid_yob, uid
###################################################################################
def ocr_wrapper_batch(ocr_dir, upload_dir):
    import os
    from datetime import datetime

    files = os.listdir(upload_dir)
    jpg_files= [ file for file in files if '.jpg' in file ]

    uid_dict={}
    filename_ls = []
    UID_NAME_ls = []
    UID_YOB_ls = []
    UID_ls = []
    start_time_ls = []
    end_time_ls = []
    for file in jpg_files:
        start_time = datetime.now()
        start_time = start_time.strftime("%d/%m/%Y %H:%M:%S")
        start_time_ls.append(start_time)
        filename, uid_name, uid_yob, uid = ocr_aadhar_base(ocr_dir, file)
        # uid_dict['FILENAME'] = filename_ls.append(filename)
        # uid_dict['UID_NAME'] = UID_NAME_ls.append(uid_name)
        # uid_dict['UID_YOB'] = UID_YOB_ls.append(uid_yob)
        # uid_dict['UID'] = UID_ls.append(uid)
        #
        filename_ls.append(filename)
        UID_NAME_ls.append(uid_name)
        UID_YOB_ls.append(uid_yob)
        UID_ls.append(uid)

        end_time = datetime.now()
        end_time = end_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_ls.append(end_time)


    uid_dict['FILENAME'] = filename_ls
    uid_dict['UID_NAME'] = UID_NAME_ls
    uid_dict['UID_YOB'] = UID_YOB_ls
    uid_dict['UID'] = UID_ls
    uid_dict['STARTTIME'] = start_time_ls
    uid_dict['ENDTIME'] = end_time_ls


    uid_df = pd.DataFrame.from_dict(uid_dict)

    return uid_df
######## Created Date : 20072023 #############################
def downloadImage(URL):
    print(f"Function Start : downloadImage ")
    """Downloads the image on the URL, and converts to cv2 RGB format"""
    from io import BytesIO
    from PIL import Image as PIL_Image
    import cv2
    import numpy as np
    import requests

    if 'http' in URL:
        response = requests.get(URL)
        image = PIL_Image.open(BytesIO(response.content))
    else:
        image = PIL_Image.open(URL)
    src = cv2.imread(URL)
    # rgb_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    rgb_img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    return rgb_img
######## Created Date : 23072023 #############################
def add_suffix(file, suffix,adddatetime):
    '''

    :param file: absolute file path
    :param suffix: suffix to be added
    :return: suffixed file path
    '''
    import os
    from datetime import datetime
    base_name, extension = os.path.splitext(file)

    if adddatetime == True:
        current_datetime = str(datetime.now().strftime("%d%m%Y_%H%M%S"))
        suffix = suffix + '_' + current_datetime
    else:
        suffix = suffix

    suffixed_file = f"{base_name}{suffix}{extension}"

    return suffixed_file
######## Created Date : 25072023 #############################
def doClassifyKyc(image, preferred_lib, conf_dict):

    log_dir = conf_dict['log_dir']
    lg.logthis(f"doClassifyKyc")
    lg.logthis(f"Classification of Text Patterns started")
    '''

    :param 
    image: image returned from downloadImage function
    preferred_lib : preferred library to classify between the KYC id (easyocr/pytesseract)
    con_dict : configurations stored in dictionary from which we will be extracting path of tesseract executable
    :return: returns tuple having matching regex matching classes
    '''
    from lib import dataConfig as dc
    import re
    import pytesseract
    import easyocr


    exe_path = conf_dict['tesseract_executable_path']
    ######## Identification of Patterns ###########################
    patterns_dict = dc.regex_dict
    pattern_set = set()

    if preferred_lib == 'easyocr':
        print(f" Pattern identification method : {preferred_lib}")
        lg.logthis(f"Pattern identification method : {preferred_lib}")
        reader = easyocr.Reader(['en'])
        ocr_out = reader.readtext(image)

        for key,vals in zip(patterns_dict.keys(),patterns_dict.values()):
            accpt_len = dc.permitted_length[key]
            # print(f"key : {key}, value : {val}")
            lg.logthis(f"key : {key}, value : {vals}, accpt_len : {accpt_len}")
            for out in ocr_out:
                out_str = out[1]
                if len(out_str) == accpt_len:
                    for val in vals:
                        lg.logthis(f"val : {val}")
                        if re.match(val,out_str):
                            pattern_set.add(key)
                            break
    elif preferred_lib == 'pytesseract':
        print(f" Pattern identification method : {preferred_lib}")
        lg.logthis(f"Pattern identification method : {preferred_lib}")

        if 'windows' in platform.platform().lower():
            pytesseract.pytesseract.tesseract_cmd = exe_path
        else:
            pass

        ocr_out = pytesseract.image_to_string(image)

        lines = ocr_out.splitlines()
        for key,vals in zip(patterns_dict.keys(),patterns_dict.values()):
            # print(f"key : {key}, value : {val}")
            accpt_len = dc.permitted_length[key]
            for line in lines:
                lg.logthis(line)
                if len(line) == accpt_len:
                    for val in vals:
                        if re.match(val,line):
                            pattern_set.add(key)
                            break

    else:
        pass
    return pattern_set
######## Created Date : 20072023 #############################
def getKycMasked_bak(image):
    '''

    :param image: image returned from downloadImage function
    :return:
    '''
    import easyocr
    import PIL
    import re
    import cv2
    import os
    ########################################################################################
    reader = easyocr.Reader(['en'])
    ocr_out = reader.readtext(image)
    #identify the PAN number
    #regex from: https://www.geeksforgeeks.org/how-to-validate-pan-card-number-using-regular-expression/
    pan_regex = "[A-Z]{5}[0-9]{4}[A-Z]{1}"
    # aadhar_regex = r"\d{4} \d{4} \d{4}"  # Regex pattern to match a 12-digit number with spaces
    aadhar_regex = r"\d{4} \d{4}"  # Regex pattern to match a 12-digit number with spaces


    for out in ocr_out:
        if re.match(aadhar_regex, out[1]):
            # print(out)
            # AADHAR=out
            #     break
            #
            # extracted the word which we need to blur and saved the it in the variable name cord
            ##cord = output[6][0]
            cord = out[0]
            # print(cord)
            # catched up the min and max the cordinates of bounding box
            x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
            x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
            # print(x_min)
            # print(y_min)
            # print(x_max)
            # print(y_max)

            # Create ROI coordinates(region of interest)
            topLeft = (x_min, y_min)
            bottomRight = (x_max, y_max)
            x, y = topLeft[0], topLeft[1]
            w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
            # print(f'width : {w}, height : {h} ')
            ## Mask prtially
            # w = round(w-(w*33/100))
            # print(f'trimmed width : {w}, height : {h} ')

            # Grab ROI with Numpy slicing and blur
            ROI = image[y:y+h, x:x+w]
            #
            blur = cv2.GaussianBlur(ROI, (51, 51), 0) # Comment for testing loop
            blur_iter=2
            for i in range(blur_iter):
                blur = cv2.GaussianBlur(blur, (51, 51), 0)


            # Insert ROI back into image
            image[y:y+h, x:x+w] = blur

            #cv2.imshow('blur', blur)
            # cv2.imshow('image', image) # commenting to avoid runtime display
            # cv2.waitKey()  # commenting to avoid runtime display

    # write to file
    # cv2.imwrite(masked_file, image) # writing file from wrapper function, hence commented
    return image
