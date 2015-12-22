import pickle
import os
import errno
import operator
import random
import os.path
import json
""" All functions in this file are not meant to surprise. If the name is not descriptive enough to know everything it does then its not in here"""

def file_exists(file_name):
    return os.path.isfile(file_name)

def read_file_to_string(file_name):
    with open(file_name, "r+") as myfile:
        data="".join(line.rstrip() for line in myfile)
    return data

def write_string_to_file(file_name, string):
    with open(file_name, "w+") as myfile:
        myfile.write(string)

def pickle_load_file_to_obj(file_name):
    with open(file_name, 'r+') as file:
        return pickle.load(file)

def pickle_dump_obj_to_file(file_name, object):
    with open(file_name, 'w+') as file:
        pickle.dump(object, file)

def read_json_file(file_name):
    return json.loads(read_file_to_string(file_name))

def write_json_file(file_name, data):
    return write_string_to_file(file_name, json.dumps(data))

def remove_dots_in_path(path):
    return path.replace('.','')

def create_path(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exception:
        pass
    return path

def add_paths(path1,path2):
    path = os.path.join(os.path.abspath(path1),os.path.basename(path2))
    create_path(path)
    return path

def create_frequency_hash(data):
    hashtable = {}
    for el in data:
        if el not in hashtable:
            hashtable[el] = 1
        else:
            hashtable[el] += 1
    return hashtable

def dict_sort_by_value(dict):
    items = dict.items()
    items.sort(key=operator.itemgetter(1))
    return items

def invert_dict(dict):
    return {val :  key for (key, val) in dict.items()}

def dict_differences(dict1,dict2):
    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    diff = {}
    for key in set1.symmetric_difference(set2):
        if key in set1:
            diff[key] = 'dict1'
        else:
            diff[key] = 'dict2'
    return diff

def chunks(n, data):
    return [data[x:x+n] for x in xrange(0, len(data), n)]

def random_interval_print(data, counter, n=100):
    if counter % 7 == 0:
        print data

def append_file_name(file_name, append):
    ext_ind = file_name.rfind('.')
    return file_name[:ext_ind] + append + file_name[ext_ind:]





