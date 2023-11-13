# img_file = r"D:\Data_Science\ds\challenges\OCR\aadhar_test1_success.jpg"

def loadData(ocr_dir, filename):
    '''

    :param ocr_dir:
    :param filename: only filename.extentsion for ex. amol.jpg
    :return:
    '''
    print('Function Start : loadData')
    # ocr_dir = r"D:\Data_Science\ds\challenges\OCR"
    image_dir = ocr_dir + "\image"
    work_dir = ocr_dir + "\work"
    config_file = ocr_dir + "\config\ocr.properties"
    # img_file = str(image_dir + '\\aadhar_test3.jpg')
    img_file = str(image_dir+'\\'+filename)
    # from PIL import Image as im
    from PIL import Image as im
    img = im.open(img_file)

    return img, img_file, image_dir, work_dir, config_file

# print("File Size In Bytes:- "+str(len(img.fp.read())))
# print( "Image shape before resizing", img.size)
# img = img.resize((900,2000))
# print( "Image shape after resizing", img.size)

