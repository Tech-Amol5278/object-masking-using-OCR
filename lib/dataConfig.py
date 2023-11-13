########## Data Configurations #############
ptrn_1 = "[A-Z]{5}[0-9]{4}[A-Z]{1}"
ptrn_2 = r"\d{4} \d{4} \d{4}"   # exact match
ptrn_3 = r"\d{4}[ .]\d{4}[ .]\d{4}" # match with exclusions in 5th and 10th Character


# regex_dict= { "pan" : ptrn_1, "aadhar" : ptrn_3 }
regex_dict= { "pan" : [ptrn_1], "aadhar" : [ ptrn_2, ptrn_3] }
permitted_length = {"pan" : 10, "aadhar" : 14}
############################################
'''
passport :
regex = "^[A-PR-WY][1-9]\\d\\s?\\d{4}[1-9]$"
/^[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]$/gm
'''
