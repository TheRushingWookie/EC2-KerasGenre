from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array,array_to_img
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint
import keras
from keras.datasets import mnist
from six.moves import range
from os import listdir
from os.path import isfile, join


from dataset_utils import *
from ec2s3 import store_to_s3, get_from_s3, get_bucket_items, shutdown_spot_request, check_for_early_shutdown, self_instance_id
import tempfile
import time
import StringIO
import requests
import theano
import os
from spectrogram import plotstft
from multi import parmap
'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).
    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''
config_file = os.environ.get('CONFIG')
config = init_config()
#config = read_json_file(config_file)

debug_mode = os.environ.get('LOCALDEBUG')
api_key = os.environ['SENDGRID']
sendgrid_url = "https://api.sendgrid.com/api/mail.send.json"

startup_data={"to" : "quinnjarr@gmail.com",
              "from" : "quinnjarr@gmail.com",
              "subject" : "Startup",
              "html" : "Starting ec2 cats vs dogs",
              }
authent_header = {"Authorization" : "Bearer %s" % api_key}
if debug_mode is None:
    requests.post(sendgrid_url, data=startup_data, headers=authent_header)
max_num_data = 300000
temp_file_name = "/tmp/genres.hdf5"
batch_size = 32
nb_epoch = 10
data_augmentation = True
# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use at each layer
nb_filters = [32, 32]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = (2, 2)
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = (3, 3)
# the CIFAR10 images are RGB
image_dimensions = 3



def load_data():

    
    config['data'] = load_files(config)
    #for song, category in config['data'].items():
    #    segment_song(song, add_paths(config['data_path'], 'segmented'))
    #song = AudioSegment.from_mp3("test1.mp3")

    #prep_spectrograms()

    x_train_imgs, x_test_imgs, y_train, y_test = gen_test_train(config)
    print("loading mp3s")

    X_train = np.zeros((len(x_train_imgs), 3, shapex, shapey), dtype="uint8")
    y_train = np.asarray(y_train, dtype="uint8")

    X_test = np.zeros((len(x_test_imgs), 3, shapex, shapey), dtype="uint8")
    y_test = np.asarray(y_test, dtype="uint8")

    for i in range(len(x_train_imgs)):
        X_train[i] = load_image(x_train_imgs[i], xsize=shapex, ysize=shapey)
    for i in range(len(x_test_imgs)):
        X_test[i] = load_image(x_test_imgs[i], xsize=shapex, ysize=shapey)


    y_train = np.reshape(y_train, (len(y_train), 1))

    y_test = np.reshape(y_test, (len(y_test), 1))

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train /= 255
    X_test /= 255
    
    return (X_train, y_train), (X_test, y_test)

# the data, shuffled and split between tran and test sets
print("Loading training data")
(X_train, y_train), (X_test, y_test)= load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, len(config['cat_to_id'].items()))
Y_test = np_utils.to_categorical(y_test, len(config['cat_to_id'].items()))

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(image_dimensions, shapey, shapex)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(config['cat_to_id'].items())))
model.add(Activation('softmax'))

print("compiling")
mode = 'FAST_COMPILE' if debug_mode else 'FAST_RUN'
#mode = 'FAST_RUN'
print("Using %s mode" % mode)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', theano_mode=mode)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.best_lost = 9239129
        self.i = 0

    def on_epoch_end(self, epoch, logs={}):
        if check_for_early_shutdown():
            save_data()
            shutdown_spot_request()
        loss = logs.get('val_loss')
        print("%s loss is %s. Current best lost is %s" % (self.i, loss, self.best_lost))
       
        if loss < self.best_lost:
            print("%s new BEST loss is %s" % (self.i, loss))
            self.i += 1
            print(self.i)
            self.best_lost = loss
            if self.i % 2 == 0:
                print("Saving data")
                save_data()
                save_email_data = startup_data
                save_email_data['html'] = "new loss is %s" % loss
                save_email_data['subject'] = "New saved!"
                if debug_mode is None:
                    requests.post(sendgrid_url, data=save_email_data, headers=authent_header)

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if check_for_early_shutdown():
            save_data()
            shutdown_spot_request()
        
def create_model():
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, len(config['cat_to_id'].items()))
    Y_test = np_utils.to_categorical(y_test, len(config['cat_to_id'].items()))

    print("starting model")
    
    model.save_weights(temp_file_name,overwrite=True)
    with open(temp_file_name) as f:
        store_to_s3(str(int(time.time())),config['s3_bucket'], f.read())
        

def load_model():
    print("Getting %s bucket items" % config['s3_bucket'])
    items = get_bucket_items(config['s3_bucket'])
    newest_item = max(map(int,items))
    print("Newest item %s, retrieving now" % newest_item)
    model_weights = get_from_s3(str(newest_item), config['s3_bucket'])
    print("loading model")
    with open('./current.weights', 'w') as f:
        f.write(model_weights)

    model.load_weights("./current.weights")
    

def save_data():
    model.save_weights(temp_file_name, overwrite=True)
    if debug_mode is None:
        with open(temp_file_name, 'r') as f:
            store_to_s3(str(int(time.time())), config['s3_bucket'], f.read())
    else:
        print("Not saving to S3, since we're in debug mode.")

def train():
    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    #checkpointer = ModelCheckpoint(filepath="/Users/quinnjarrell/Desktop/Experiments/keras/saved/", verbose=1, save_best_only=True)
    history = LossHistory()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=100, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),callbacks=[history])
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    if debug_mode is None:
        save_data()
        shutdown_spot_request()
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    import matplotlib.pyplot as plt

    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)
    plt.show()

def predict():
    import pylab as pl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    np.set_printoptions(precision=5, suppress=True)
    #convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
    predictions = model.predict(X_test[:10]).tolist()
    print("Tests ", zip(predictions, X_test[:10].tolist()))
    import pdb; pdb.set_trace()  # breakpoint 8d9fb711 //

    Y_pred = model.predict(X_test)
    # Convert one-hot to index
    y_pred = np.argmax(Y_pred, axis=1)
    from sklearn.metrics import classification_report
    print(classification_report(y_test , y_pred))
    # Visualize convolution result (after activation)
    i = 100
    # Visualize the first layer of convolutions on an input image
    X = X_test[i:i+1]
    print("I predict a ", model.predict(X))
    
    plt.figure()
    plt.title('input')

    nice_imshow(plt.gca(), array_to_img(np.squeeze(X)), vmin=0, vmax=1, cmap=cm.binary)
    #C1 = convout1_f(X)
    #C1 = np.squeeze(C1)
    #print("C1 shape : ", C1.shape)

    plt.figure(figsize=(shapex, shapey))
    plt.suptitle('convout1')
    #nice_imshow(plt.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)

def predict_image(image_path, img_type):

    for i in range(1):
        data = load_image(image_path, need_augments=True)

        print(hash(tuple(data.tolist()[0][0])))
        type_id = type_name_to_type_id[img_type]
        X_pred = np.zeros((1, 3, 32, 32), dtype="uint8")
        X_pred[0] = data
        X_pred = X_pred.astype("float32")
        X_pred /= 255
        prediction = model.predict(X_pred)
        pred_max = np.argmax(prediction[0])
        pred_name = config['categories'][pred_max]
        pred_conf = prediction[0][pred_max]


        print(prediction)
        print("Guessing %s is a %s with confidence %s. Guess is %s" % (image_path, pred_name, pred_conf, pred_name == img_type))


def save_graph():
    from keras.utils.visualize_util import plot
    plot(model, to_file=str(time.time()) + "-" + self_instance_id + '.png')



create_model()
#load_model()
#train()
#predict()
#predict_image("/Users/quinnjarrell/Downloads/catbread.jpg", 'cat')
#save_graph()
