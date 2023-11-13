from PIL import Image, ImageDraw
from lib import functions as fnc
from lib import load_configs as cf
from lib.generic import logger as lg
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
from lib import dataConfig as dc
from lib.dataConfig import regex_dict as rgx
import easyocr
import PIL
import re
import cv2
import os
######################################################################################
mask_style = cf.mask_style.lower()
#######################################################################################
def getDPI(img_file):
    print('Function Start : getDPI')
    from PIL import Image

    img = Image.open(img_file)
    img_info = img.info
    # get the current dpi of image
    input_dpi = img_info.get('dpi', (72,72))

    return input_dpi
#######################################################################################
def get_binary_image(img_file,method):
    print('Function Start : get_binary_image')
    import cv2

    ''' 
    Binarization Methods
        1. otsu : Otsu’s Binarization
        2. adapt_thres : Adaptive Thresholding
    '''
#     load the image
    print('img_file', img_file)
    # img = cv2.imread(img_file,0)
    img = cv2.imread(img_file)
#     check if img is loaded successfully
    if img is None:
        return "Failed to load image"
    else:
#     start binarization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if method == 'otsu' :
            ret, imgf = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU) #imgf contains Binary image
        elif method == 'adapt_thres':
            imgf = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #imgf contains Binary image
        else:
            pass

#         Display thresholded image
#         cv2.imshow("Thresholded image",imgf)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.bitwise_not(gray)

        # ####### process to remove salt and pepper noise ###############################
        # # Apply median filtering with a specified kernel size (e.g., 3x3)
        # median_filtered = cv2.medianBlur(imgf, 11)
        return imgf;
#######################################################################################
def doSkewCorrection(bin_img, method):
    '''

    :param bin_img:
    :param method:
                projection_profile -- Projection profile method, widely used
                Hough transformation method
                Topline method
                Scanline method
            '''
    if method.lower() == "projection_profile":
        import numpy as np
        from PIL import Image as im
        from scipy import ndimage as inter

        ###########################################################
        lg.logthis("Skew correction started")
        img = im.open(bin_img)
        wd, ht = img.size
        pix = np.array(img.convert('1').getdata(), np.uint8)
        bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score


        delta = 1
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(bin_img, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        lg.logthis(f"angles = {angles}")
        lg.logthis(f"scores = {scores}")
        lg.logthis(f"Best Score {best_score}")
        lg.logthis(f"Best Angle {best_angle}")
        lg.logthis(f"Median Angle {np.median(angles)}")
        print('Best angle: {}'.format(best_angle))
        # correct skew
        data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
        skewCorrected = im.fromarray((255 * data).astype("uint8")).convert("RGB")
        # img.save('skew_corrected.png')
    else:
        pass
    return skewCorrected

#######################################################################################
def pdfTojpg(input_pdf_path, output_dir):
    import os
    from PyPDF2 import PdfFileReader
    from PIL import Image
    with open(input_pdf_path, "rb") as pdf_file:
        pdf_readr = PdfFileReader(pdf_file)
        num_pages = pdf_readr.numPages

    # create the output folder if not exists already
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for page_num in range(num_pages):
        pdf_page = pdf_readr.getPage(page_num)
        image = pdf_page.extractText()
        image_path = os.path.join(output_dir, f"convertFromPdf_{page_num + 1}.jpg")

    # save the image in JPEG format
    image.save(image_path, "JPEG")

    return
##################################################################
def correct_skew(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(blurred, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     print(lines)

    # Calculate the angle of each line segment
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)
#         print(line[0], x1, y1, x2, y2, angle)


    # Calculate the median angle
    median_angle = np.median(angles)
    print(median_angle)

    if np.abs(median_angle) > 1:
        print(np.abs(median_angle))
        # Rotate the image to correct the skew
        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), np.degrees(median_angle), 1), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return rotated
    else:
        return image

    # Rotate the image to correct the skew
#     rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), np.degrees(median_angle), 1), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return rotated
########################################
def correct_skew_test_1(image_path, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
          borderMode=cv2.BORDER_REPLICATE)
    rotated_img = cv2.imwrite('rotated.png', rotated)
    return best_angle, rotated
########################################################3
def correct_skew_with_hough_lines_test_2(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection (optional)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Calculate the skew angle based on the detected lines
    skew_angle = 0.0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)  # Convert angle from radians to degrees
            skew_angle += angle
        skew_angle /= len(lines)  # Average angle

    # Rotate the image to correct the skew
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), skew_angle, 1), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return rotated
######## Created Date : 20072023 #############################
def getKycMasked(image,input_file, patterns):
    print(f"Function Start : getKycMasked ")
    '''

    :param image: image returned from downloadImage function
    :return:
    '''
    from lib.dataConfig import regex_dict as rgx
    import easyocr
    import PIL
    import re
    import cv2
    import os
    ########################################################################################
    image = cv2.imread(input_file)
    reader = easyocr.Reader(['en'])
    ocr_out = reader.readtext(image)
    #identify the PAN number
    #regex from: https://www.geeksforgeeks.org/how-to-validate-pan-card-number-using-regular-expression/
    # pan_regex = "[A-Z]{5}[0-9]{4}[A-Z]{1}"
    # # aadhar_regex = r"\d{4} \d{4} \d{4}"  # Regex pattern to match a 12-digit number with spaces
    # aadhar_regex = r"\d{4} \d{4}"  # Regex pattern to match a 12-digit number with spaces

    for ptn in patterns:
        lg.logthis(f"patterns {ptn}")
        for key, vals in zip(rgx.keys(),rgx.values()):
            lg.logthis(f"key {key}, vals {vals}, ptn, {ptn}")
            accpt_len = dc.permitted_length[key]
            if ptn == key:
                for val in rgx[key]:
                    lg.logthis(f"val {val}")
                    # _rgx = rgx[val]
                    _rgx = val
                    lg.logthis(f"ocr_out, {ocr_out}")
                    match_cnt=0
                    match_dict={}
                    for out in ocr_out:
                        if len(out[1]) == accpt_len:
                            match_dict[out[1]] = 0
                            lg.logthis(f"match_dict, {match_dict}")
                            lg.logthis(f"Check pattern {val} if matching with {out[1]}")
                            ######################################
                            lg.logthis(f"_rgx {_rgx}")
                            if match_dict[out[1]] == 0 :
                                if re.match(_rgx, out[1]):
                                    match_dict[out[1]] +=1
                                    lg.logthis(f"pattern {val} matched with {out[1]}")
                                    match_cnt += 1
                                    lg.logthis(f"ROI >> {out[1]}")
                                    ###### match percentage ##############
                                    match_obj = re.match(_rgx, out[1])
                                    match_percent = (len(match_obj.group()) / len(out[1])) * 100
                                    lg.logthis(f"match_obj >> {match_obj}")
                                    lg.logthis(f"matched >> {match_percent} %")
                                    ######################################
                                    # extracted the word which we need to blur and saved the it in the variable name cord
                                    cord = out[0]
                                    lg.logthis(f"cord >> {cord}")
                                    ## catched up the min and max the coordinates of bounding box
                                    x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
                                    x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
                                    ## Create ROI coordinates(region of interest)
                                    topLeft = (x_min, y_min)
                                    bottomRight = (x_max, y_max)
                                    x, y = topLeft[0], topLeft[1]
                                    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                                    ## Mask prtially
                                    # w = round(w-(w*33/100))
                                    # print(f'trimmed width : {w}, height : {h} ')
                                    # Grab ROI with Numpy slicing and blur
                                    ROI = image[y:y+h, x:x+w]
                                    # blur = cv2.GaussianBlur(ROI, (51, 51), 0)
                                    # blur_iter=3
                                    # ## Stacking up gaussian blur to thicken effect
                                    # for i in range(blur_iter):
                                    #     blur = cv2.GaussianBlur(blur, (51, 51), 0)
                                    if mask_style=='blur':
                                        lg.logthis("Selected  Blurring Effect")
                                        blur = doGaussianBlur(ROI,3)
                                        # Insert ROI back into image
                                        image[y:y+h, x:x+w] = blur
                                    else:
                                        lg.logthis("Selected Darkening Effect")
                                        # input_file = cv
                                        image = Image.open('your_image.jpg')
                                        # Create a drawing context
                                        draw = ImageDraw.Draw(image)
                                        # Define the area to cover with a black box (left, top, right, bottom)
                                        black_box_area = (100, 100, 300, 300)  # Adjust these coordinates as needed
                                        # black_box_area = (x_min, y_min,x_max, y_max)
                                        # Draw a black rectangle over the specified area
                                        draw.rectangle(black_box_area, fill="black")
                                        image.save(r"D:\Data_Science\ds\challenges\OCR\kyc_masking\out\testing_dark_effect.jpg")


                        #cv2.imshow('blur', blur)
                        # cv2.imshow('image', image) # commenting to avoid runtime display
                        # cv2.waitKey()  # commenting to avoid runtime display

    # write to file
    # cv2.imwrite(masked_file, image) # writing file from wrapper function, hence commented
    # Convert the binary image to an RGB image
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lg.logthis(f"Final match_dict, {match_dict}")
    return image
########## 131023 #########################################
def doGaussianBlur(ROI, iter):
    '''
    :param ROI: image object or region of interest
    :param iter: number of times the function should perform iter
    :return: image object

    '''
    lg.logthis(f"Performing Gaussian Blur for ROI {ROI}")
    iter = iter-1
    blur = cv2.GaussianBlur(ROI, (51, 51), 0) # 1st iteration
    ## Stacking up gaussian blur to thicken effect
    for i in range(iter): # iteration in loop
        blur = cv2.GaussianBlur(blur, (51, 51), 0)

    lg.logthis("Blurring process ended")
    return blur
################### 17102023 ###################################
def getKycMasked_new(input_file):
    '''

    :param input_file:
    :param patterns:
    :return:
    '''
    docs = dc.regex_dict.keys()
    ######### Read Image ###########
    image = cv2.imread(input_file)
    reader = easyocr.Reader(['en'])
    ocr_out = reader.readtext(image)
    ################################
    lg.logthis(f"Acquiring possible targets")
    docs_matched=set()
    for doc in docs:
        ptns = dc.regex_dict[doc]
        max_len = dc.permitted_length[doc]
        param_str = f"doc = {doc}, ptns = {ptns}, max_len = {max_len}"
        lg.logthis(param_str)
        pos_targets = {}
        for out in ocr_out:
            cord,pos_out = out[0],out[1]
            # cord_ls = []
            lg.logthis(f"pos_out = {pos_out}, len = {len(pos_out)}, cord = {cord}")
            if len(pos_out) == max_len:
                # lg.logthis(f"possible target found {pos_out} using params {param_str}")
                # pos_targets[pos_out] = cord
                if pos_out not in pos_targets:
                    cord_ls = []
                    cord_ls.append(cord)
                    pos_targets[pos_out] = cord_ls
                else:
                    lg.logthis(f"pos_out {pos_out} already present in pos_targets, hence appending to its coordinates")
                    lg.logthis(f"pos_targets = {pos_targets}")
                    lg.logthis(f"existing val for pos_out  = {pos_targets[pos_out]} , {type(pos_targets[pos_out])}")
                    cord_ls = pos_targets[pos_out]
                    cord_ls.append(cord)
                    pos_targets[pos_out] = cord_ls
            else:
                lg.logthis(f"possible target not found")
        lg.logthis(f"pos_targets = {pos_targets} using params {param_str}")
        ######################################################################
        final_target={}
        # docs_matched=set()
        lg.logthis(f"final_target initialized")
        for pos_out,cord in pos_targets.items():
            for ptn in ptns:
                # lg.logthis(f"working for ptn {ptn}")
                if pos_out not in final_target:
                    if re.match(ptn, pos_out):
                        final_target[pos_out] = cord
                        docs_matched.add(doc)
                        # lg.logthis(f"match found for {ptn} and {pos_out}")
                    else:
                        pass
                        # lg.logthis(f"no match found for {ptn} and {pos_out}")
                else:
                    lg.logthis(f"pos_out already in final_target")
        ######################################################################
        lg.logthis(f"final_target = {final_target} using params {param_str}")
        if final_target.items():
            for fin_out,cords in final_target.items():
                lg.logthis(f"fin_out {fin_out}, cords {cords}")
                for cord in cords:
                    lg.logthis(f"cord {cord}")
                    ## catched up the min and max the coordinates of bounding box
                    x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
                    x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
                    ## Create ROI coordinates(region of interest)
                    topLeft = (x_min, y_min)
                    bottomRight = (x_max, y_max)
                    x, y = topLeft[0], topLeft[1]
                    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                    # Grab ROI with Numpy slicing and blur
                    ROI = image[y:y+h, x:x+w]
                    #################################
                    if mask_style=='blur':
                        lg.logthis("Selected  Blurring Effect")
                        blur = doGaussianBlur(ROI,3)
                        # Insert ROI back into image
                        image[y:y+h, x:x+w] = blur
                        ############### Convert to RGB Image #############################
                        unique_values = len(np.unique(image))
                        if unique_values == 2:
                            image_type = "Binary"
                        elif unique_values <= 256:
                            image_type = "Grayscale"
                        else:
                            image_type = "Color"
                        lg.logthis(f"Image Type {image_type}")

                        # Load the binary image (grayscale, where values are typically 0 and 255)
                        # binary_image = cv2.imread('binary_image.jpg', cv2.IMREAD_GRAYSCALE)

                        # Define the color for the foreground (where binary image is 255)
                        # foreground_color = (0, 0, 255)  # In BGR format (red)

                        # Create an all-zero (black) image with the same dimensions as the binary image
                        # rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

                        # Set the pixels corresponding to the foreground in the RGB image to the specified color
                        # rgb_image[image == 255] = foreground_color

                        # Save the RGB image
                        # cv2.imwrite('rgb_image.jpg', rgb_image)
                        ######################################################################################
                        # Convert grayscale to RGB
                        # rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        ######################################################################################
                    else:
                        lg.logthis("Selected Darkening Effect")
        else:
            lg.logthis("No item exist in final target")
        ######################################################################


    return image, docs_matched
    # return rgb_image, docs_matched
######################### 17102023 ####################################################
def doPatternCheck(input_file):
    '''

    :param input_file:
    :return:
    '''

    lg.logthis(f"Doing documents screening")
    docs = dc.regex_dict.keys()
    image = cv2.imread(input_file)
    reader = easyocr.Reader(['en'])
    ocr_out = reader.readtext(image)
    ################################
    lg.logthis(f"Acquiring possible targets")
    for doc in docs:
        ptns = dc.regex_dict[doc]
        max_len = dc.permitted_length[doc]
        param_str = f"doc = {doc}, ptns = {ptns}, max_len = {max_len}"
        lg.logthis(param_str)
        pos_targets = {}
        for out in ocr_out:
            cord,pos_out = out[0],out[1]
            if len(pos_out) == max_len:
                pos_targets[pos_out] = cord
        lg.logthis(f"pos_targets = {pos_targets} using params {param_str}")
        ######################################################################
        final_target={}
        patterns_matched = set()
        lg.logthis(f"final_target initialized")
        for pos_out,cord in pos_targets.items():
            for ptn in ptns:
                lg.logthis(f"working for ptn {ptn}")
                if pos_out not in final_target:
                    if re.match(ptn, pos_out):
                        final_target[pos_out] = cord
                        patterns_matched.add(doc)
                        # lg.logthis(f"match found for {ptn} and {pos_out}")
                    else:
                        pass
                        # lg.logthis(f"no match found for {ptn} and {pos_out}")



    return patterns_matched
####################3 18102023 ###############################################################
def get_binary_image_test(img_file,method):
    print('Function Start : get_binary_image')
    import cv2

    ''' 
    Binarization Methods
        1. otsu : Otsu’s Binarization
        2. adapt_thres : Adaptive Thresholding
    '''
#     load the image
    print('img_file', img_file)
    # img = cv2.imread(img_file,0)
    img = cv2.imread(img_file)
#     check if img is loaded successfully
    if img is None:
        return "Failed to load image"
    else:
#     start binarization
        ######### convert to gray ##########################
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
        #######################################################
        if method == 'otsu' :
            ret, imgf = cv2.threshold(enhanced_image, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU) #imgf contains Binary image
        elif method == 'adapt_thres':
            imgf = cv2.adaptiveThreshold(enhanced_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #imgf contains Binary image
        else:
            pass

        return imgf;
############## 19102023 ###################################################
def gray2rgb(input):

    # Load the grayscale image
    gray_image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)

    # Convert grayscale to RGB
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    return rgb_image
