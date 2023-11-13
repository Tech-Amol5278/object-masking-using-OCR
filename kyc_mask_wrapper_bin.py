import sys
from lib.generic import fileProcessing as fp
from lib import load_configs as cf
from lib import functions as fnc, imageProcessing as imp
from lib.generic import fileProcessing as fp, logger as lg, dbProcess as db
import os
import cv2
import sys
from datetime import datetime
################################################################################
'''

'''
################################################################################
def main_func(mode, *args):

   ######### load temp vars ######################
    backup_parent_dir = cf.backup_parent_dir
    sfx_rgb = cf.sfx_rgb
    work_dir = cf.work_dir
    searched_str_classification_method = cf.searched_str_classification_method
    sfx_skwd = cf.sfx_skwd
    sfx_mask = cf.sfx_mask
    sfx_bin = cf.sfx_bin
    sfx_mask_compressed = cf.sfx_mask_compressed
    output_dir = cf.output_dir
    image_quality = int(cf.image_quality)
    ##########################################################
    i = 0
    ip_list = fp.getFiles(sys.argv)
    ######## Get Batch ID #####################################
    mode=mode.upper()
    if mode == 'BATCH':
        '''
            Batch ID from databases for unique reference
        '''
        db_engine = cf.db_config_dict['db_engine']
        if db_engine == 'MYSQL':
            pass
        else:
            batch_id = db.doFetch("SELECT NEXT VALUE FOR SEQ_BATCH_ID")[0]
        lg.logthis(f"Processing Batch : {batch_id}")
    else:
        lg.logthis(f"API Mode")
    ###########################################################################################
    if len(ip_list) >= 1:
        # exec = True
        # if exec == False:
        #     pass
        # else:
        ######## get inputs from sys ############################
        # print(f"length of args : {len(sys.argv)}")
        # print(f"print all args : {sys.argv} ")
        # print(f"Number of Arguments passed : {len(sys.argv)-1}")
        ########################################################
        for i in ip_list:
            ######## Get Request ID #####################################
            '''
                Request ID from databases for unique reference
            '''
            db_engine = cf.db_config_dict['db_engine'].lower()
            if db_engine == 'mysql':
                last_req_id = db.doFetch("SELECT MAX(REQUEST_ID) FROM TB_SEQUENCE")[0]
                db_insert_str = f"INSERT INTO TB_SEQUENCE(REQUEST_ID) VALUES({last_req_id+1})"
                db.doInsert(db_insert_str)
                lg.logthis(db_insert_str)
                req_id = db.doFetch("SELECT MAX(REQUEST_ID) FROM TB_SEQUENCE")[0]
                pass
            else:
                req_id = db.doFetch("SELECT NEXT VALUE FOR SEQ_REQUEST_ID")[0]
            lg.logthis(f"Processing Request : {req_id}")
            ######### Count of Total Processes ##########################
            total_prcs = 5
            ######### Backup of Input File ##############################
            input_file = i
            print(f"Input File >> {input_file}")
            curr_date = datetime.now()
            curr_date = str(curr_date.strftime("%d%m%Y"))
            backup_child_dir = os.path.join(backup_parent_dir,curr_date)

            if not os.path.exists(backup_child_dir):
                os.makedirs(backup_child_dir)
                print(f" Directory {backup_child_dir} created successfully")
                lg.logthis(f"Directory {backup_child_dir} created successfully")
            else:
                print(f" Directory {backup_child_dir} already exists")
                lg.logthis(f"Directory {backup_child_dir} already exists")


            prcs_pending_cnt = total_prcs-1
            prcs_pending_msg = f" {prcs_pending_cnt} processes pending."
            prc_status = 1

            if mode.lower() == 'batch':
                db_insert_str = f"INSERT INTO TB_REQUEST_DETAILS (REQUEST_ID, INPUT,LAST_UPDATE, STATUS,BATCH_ID,MODE) " \
                                f"VALUES ({req_id}," \
                                f"{fp.doInvertedEnclose(input_file)}, " \
                                f"'Backup Step Skipped, {prcs_pending_msg}', " \
                                f"{prc_status}, " \
                                f"{batch_id}, " \
                                f"'{mode}')"
            else:
                db_insert_str = f"INSERT INTO TB_REQUEST_DETAILS (REQUEST_ID, INPUT,LAST_UPDATE, STATUS,MODE) " \
                                    f"VALUES ({req_id}," \
                                    f"{fp.doInvertedEnclose(input_file)}, " \
                                    f"'Backup Step Skipped, {prcs_pending_msg}', " \
                                    f"{prc_status}, " \
                                    f"'{mode}')"

            lg.logthis(db_insert_str)
            db.doInsert(db_insert_str)
            ######## Backup Directory created ##########################
            ######## Generate random name for input file ###############
            random_name = datetime.now()
            random_name = str(random_name.strftime("%d%m%Y%H%M%S"))
            print(random_name)
            lg.logthis(random_name)
            ##############################################################
            # if 'input_file' in globals():
            print(f"Processing file : {input_file} ")
            lg.logthis(f"Processing file : {input_file} ")
            #########################################################
            ip_file = os.path.basename(input_file)
            image = cv2.imread(input_file)
            ######### get Binary Image #################################
            lg.logthis(f"Binary conversion in process")
            # rgb_img = fnc.downloadImage(input_file)
            bin_img = imp.get_binary_image(input_file, "adapt_thres")
            # bin_img = imp.get_binary_image_test(input_file, "adapt_thres")
            bin_name = fnc.add_suffix(ip_file, sfx_bin, adddatetime=False)

            bin_file = os.path.join(work_dir, bin_name)
            cv2.imwrite(bin_file, bin_img)
            print(f"Binary converted file saved to : {bin_file} ")
            lg.logthis(f"Binary converted file saved to : {bin_file}")

            prcs_pending_cnt = prcs_pending_cnt-1
            prcs_pending_msg = f" {prcs_pending_cnt} processes pending."
            prc_status += 1

            db_updt_str = f"update TB_REQUEST_DETAILS set STATUS={prc_status}, " \
                          f"LAST_UPDATE = 'Binary Conversion Completed, {prcs_pending_msg}'," \
                          f"PROCESSED_TIME = {db.now}" \
                          f" where REQUEST_ID = {req_id}"

            lg.logthis(db_updt_str)
            db.doInsert(db_updt_str)
            lg.logthis("Binary Conversion Completed")
            ######## Skew_Correction ################################
            dskewd_img = imp.doSkewCorrection(bin_file,"projection_profile")

            # dskewd_img = imp.correct_skew(bin_file)
            # dskewd_img = imp.correct_skew_test_1(input_file, delta=1, limit=5)[1]
            # dskewd_img = imp.correct_skew_with_hough_lines_test_2(input_file)
            dskewd_name = fnc.add_suffix(ip_file, sfx_skwd, adddatetime=False)
            dskewd_file = os.path.join(work_dir, dskewd_name)
            dskewd_img.save(dskewd_file)
            # cv2.imwrite(dskewd_file, dskewd_img)

            prcs_pending_cnt = prcs_pending_cnt-1
            prcs_pending_msg = f" {prcs_pending_cnt} processes pending."
            prc_status += 1
            db_updt_str = f"update TB_REQUEST_DETAILS set STATUS={prc_status}, " \
                          f"LAST_UPDATE = 'Skew Correction Completed, {prcs_pending_msg}'," \
                          f"PROCESSED_TIME = {db.now}" \
                          f" where REQUEST_ID = {req_id}"
            lg.logthis(db_updt_str)
            db.doUpdate(db_updt_str)
            lg.logthis(f"Skew corrected file saved to : {dskewd_file} ")
            ########################################################
            ######### Blurring effect ################################
            masked_name = fnc.add_suffix(ip_file, sfx_mask, adddatetime=False)
            masked_file = os.path.join(work_dir, masked_name)
            # masked_img  = fnc.getKycMasked(rgb_img)
            # masked_img  = imp.getKycMasked(image, dskewd_file, docs_match_set) # rgb_img
            masked_img,docs_matched  = imp.getKycMasked_new(dskewd_file)
            cv2.imwrite(masked_file, masked_img)
            lg.logthis(f"docs_matched = {docs_matched}")

            pats = ''
            for pat in docs_matched:
                print(pat)
                if len(pats) >0:
                    print(f"pattens length >> {len(pats)}")
                    pats = pats + ', ' + pat
                else:
                    pats = pats + pat
            #####################################################
            prcs_pending_cnt = prcs_pending_cnt-1
            prcs_pending_msg = f" {prcs_pending_cnt} processes pending."
            prc_status += 1
            #####################################################
            if len(docs_matched) == 0:
                db_updt_str = f"update TB_REQUEST_DETAILS set STATUS={prc_status}, " \
                              f"LAST_UPDATE = 'No documents found'," \
                              f"PROCESSED_TIME = {db.now}," \
                              f"PATTERNS='NA'" \
                              f" where REQUEST_ID = {req_id}"
                lg.logthis(db_updt_str)
                db.doUpdate(db_updt_str)
                lg.logthis(f"No documents found, further processing stopped")
            else :
                db_updt_str = f"update TB_REQUEST_DETAILS set STATUS={prc_status}, " \
                              f"LAST_UPDATE = 'Pattern Masking Completed, {prcs_pending_msg}'," \
                              f"PROCESSED_TIME = {db.now}," \
                              f"PATTERNS={fp.doInvertedEnclose(pats)}" \
                              f" where REQUEST_ID = {req_id}"
                lg.logthis(db_updt_str)
                db.doUpdate(db_updt_str)
                lg.logthis(f"KYC masked and saved to : {masked_file}, proceeding further for compression")
                ######## Compression ####################################
                masked_compressed_name = fnc.add_suffix(ip_file, sfx_mask_compressed, adddatetime=True)
                masked_compressed_file = os.path.join(output_dir, masked_compressed_name)
                _masked_img = cv2.imread(masked_file)
                # lg.logthis(f"masked_file >> {masked_file}")
                # lg.logthis(f"masked_compressed_file >> {masked_compressed_file}")
                # lg.logthis(f"image_quality >> {image_quality}")
                # lg.logthis(f"_masked_img >> {_masked_img}")
                if _masked_img is not None:
                    print("_masked_img exists")
                    cv2.imwrite(masked_compressed_file, _masked_img, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
                    print(f"File compressed and saved to {masked_compressed_file}, Thank you :)")

                prcs_pending_msg = ' No process pending.'
                prc_status += 1
                db_updt_str = f"update TB_REQUEST_DETAILS set STATUS={prc_status}, " \
                              f"OUTPUT = {fp.doInvertedEnclose(masked_compressed_file)}, " \
                              f"LAST_UPDATE = 'Image Compression Completed, {prcs_pending_msg}'," \
                              f"PROCESSED_TIME = {db.now}" \
                              f" where REQUEST_ID = {req_id}" \

                lg.logthis(db_updt_str)
                db.doUpdate(db_updt_str)
                lg.logthis(f"File compressed and saved to {masked_compressed_file}, Thank you :)")

    else :
        print("Incorrect number of Arguments")
        lg.logthis("Incorrect number of Arguments")
    return


ip_list = fp.getFiles(sys.argv)
if len(ip_list) == 1 :
    sys.stdout.flush()  # flush the output buffer
    main_func('API', ip_list)
else :
    # sys.stdout.flush()  # flush the output buffer
    main_func('BATCH', ip_list)

