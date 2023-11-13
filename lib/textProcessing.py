from lib import functions as fnc

def doTextProcess(str):
    print('Function Start : doTextProcess')
    name_pos = 8
    uid_pos = 26
    contact_pos = 21

   ############# Address formation ######################
    addr_line1 = str.splitlines()[11]
    addr_line2 = str.splitlines()[12]

    addr = "".join([addr_line1, addr_line2])
    ############################################################
    #######################################
    uid_info = {}
#     uid_info['name'] = str.partition('\n')[name_pos]
#     uid_info['uid'] = str.partition('\n')[25]
    uid_info['name'] = str.splitlines()[name_pos]
    uid_info['uid'] = str.splitlines()[uid_pos]
    uid_info['contact'] = str.splitlines()[contact_pos]
    uid_info['address'] = addr


#     uid_pos = 25
    return uid_info

def doClean(extr_file, noext, work_dir):
    print('Function Start : doClean')
    '''
        Remove blank lines
    '''
    # Read extracted file and create another file ".cleaned"
    clean_file = work_dir+'\\'+noext+'.cleaned'
    print('clean_file : ', clean_file)
    print('extr_file : ', extr_file)
    with open(extr_file, 'r') as in_file, open(clean_file, 'w') as out_file:
        for line in in_file:
            if not line.isspace():
                out_file.write(line)

        # # check for word "Enrollment"
        # for pos, line in enumerate(lines, 0):
        #     if 'Enrollment' in line:
        #         enrol_pos = pos
        #     else:
        #         pass
        #
        # # get the string at the above index/position
        # with open(ext_file, 'r') as file:
        #     u_name = file.readline(enrol_pos)

    return clean_file



def getIndex(textfile, search_word):
    with open(textfile,'r') as file:
        lines = file.readlines()

    for pos, line in enumerate(lines):
        if search_word in line:
            word_line_index = pos
            search_line = line
            break
    return word_line_index, search_line


def getStrForward(clean_file, search_word,add_pos):
    print('Function Start : getStrForward')
    '''
        Logic : Based on Enrollment no
        Input :
            clean_file : filepath of cleaned file(removed blank lines)(String)
            search_word : word to be searched in file, only 1st occurance will be traced(String)
            add_pos : position to be added to get the exact line basis on the searched word(Int)
    '''
    # word_line_index, search_line = getIndex(clean_file,'Enrollment')
    word_line_index, search_line = getIndex(clean_file, search_word)
    target_line_index = word_line_index+add_pos

    with open(clean_file,'r') as file:
        lines = file.readlines()
        target_line = lines[target_line_index].strip()

    return target_line, target_line_index


def getYob(clean_file, search_word,minus_pos):
    print('Function Start : getYob')

    word_line_index, search_line = getIndex(clean_file,search_word)
    yob_line_index = word_line_index-minus_pos

    with open(clean_file,'r') as file:
        lines = file.readlines()
        u_yob = lines[yob_line_index].strip()

    return u_yob, yob_line_index


