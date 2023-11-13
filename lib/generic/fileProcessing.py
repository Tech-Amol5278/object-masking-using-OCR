'''
General file processing functions, to make the tasks easier

'''
def getFiles(*args):
    import sys
    '''
    Function to check the number of arguements passed to the python main function
    :param args:
    :return:
    '''
    file_list = [i for i in sys.argv if ('.py' or '.exe') not in i ]

    return file_list

class doDir:
    from datetime import datetime
    import os
    def __init__(self,dir,name):
        self.dir = dir


    def createDir(self):
        curr_dt = self.datetime.now()
        curr_dt = str(curr_dt.strftime("%d%m%Y%H%M%S"))
        curr_dir = self.os.path.join(dir,curr_dt)

        if not self.os.path.exists(curr_dir):
            self.os.makedirs(curr_dir)
            print(f" Directory {curr_dir} created successfully")
        else:
            print(f" Directory {curr_dir} already exists")


def doInvertedEnclose(str):
    return "'"+str+"'"
