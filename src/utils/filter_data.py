import yaml 
import re

def load_regex(filename):
    regex_list = []
    with open(filename, 'r') as val:
        document = yaml.safe_load(val)
        regex_list = document
    return regex_list

# reg2 = regex_list[6].get('regex').strip()
# print(repr(reg2)) #print the raw string

def apply_regex_to_string(regex_list, string):
    new_string = string
    for rex in regex_list:
        reg = rex.get('regex').strip()
        raw_s = r'{}'.format(reg)
        if re.search(raw_s, string):
            new_string = re.sub(raw_s, rex.get('code') + " ", string)
            break
    return new_string

def main():
    pass

if __name__ == "__main__":
    main()