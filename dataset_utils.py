from pydub import AudioSegment
from utilities import *
from sklearn.cross_validation import train_test_split
import json
import glob
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array,array_to_img
debug = os.environ.get('LOCALDEBUG')

data_path = os.environ.get('DATA_PATH')


def init_config():
    config = {}
    config['data_path'] = data_path 
    config['id_to_cat'] = dict(enumerate(['alternative', 'funksoulrnb', 'raphiphop', 'blues', 'folkcountry', 'pop', 'electronic', 'jazz','rock']))


    config['cat_to_id'] = invert_dict(config['id_to_cat'])
    config['training'] = {'path' : add_paths(data_path, "training"), 'files' : None}
    config['testing'] = {'path' : add_paths(data_path, "testing"), 'files' : None}
    config['s3_bucket'] = os.environ.get("BUCKET")
    config['data'] = None
    return config

def convert_song_to_img(x_train, x_test):
    #BROKEN FOR NOW
    x_train = parmap(lambda file_name: (file_name, AudioSegment.from_mp3(file_name)), x_train)
    x_test = parmap(lambda file_name: (file_name, AudioSegment.from_mp3(file_name)), x_test)
    print("converting to spectrograms")
    song = None
    x_train = map(lambda song : plotstft(song[1].get_array_of_samples(), song[1].frame_rate, song[0]), x_train)

    x_test = map(lambda song : plotstft(song[1].get_array_of_samples(), song[1].frame_rate, song[0]), x_test)
    return x_train, x_test

def load_files(config):
    data = {}
    for folder in os.walk(data_path):
        if folder[0] == data_path:
            continue
        category = folder[0][folder[0].rfind('/') + 1:]
        for file_name in folder[2]:
            if file_name == '.DS_Store':
                continue
            #print folder[0], file_name
            data[folder[0] +'/' + file_name] = config['cat_to_id'][category]
    return data

def gen_test_train(config):
    return train_test_split(config['data'].keys(), config['data'].values())

def load_image(image_path, xsize=32, ysize=32, need_augments=False):
    img = Image.open(image_path)
    img = img.convert("RGB")

    if need_augments:
        img = img.resize((xsize,ysize))
    data = img_to_array(img)
    img.close()
    return data

def segment_song(file_name, export_path):
    song = AudioSegment.from_mp3(file_name)
    #Pydub length is in milliseconds. Songs are 10 secs long to divide into 3 and lose last milliseconds
    segments = chunks(3333, song)[:-1]

    for idx, segment in enumerate(segments):
        path = export_path + file_name
        create_path(path)
        segment.export(append_file_name(path, str(idx)))
