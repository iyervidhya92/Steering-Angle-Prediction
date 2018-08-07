import glob
import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity

import argparse
import base64
import json

import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras import backend as K

class Model(object):
    def __init__(self, 
                 model_path,
                 X_train_mean_path):
        
        self.model = load_model(model_path)
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)
        
    def predict(self, img_path):
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)
        
        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle
            
        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1
            
            return self.mean_angle
            
        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1
            
            X = np.concatenate(self.state, axis=-1)
            X = X[:,:,::-1]
            X = np.expand_dims(X, axis=0)
	    # print (X.shape)
	    # print (self.X_mean.shape)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
            
            return self.model.predict(X)[0]
        
#---------------------------------------------------------------------------------------------------------------------

K.set_image_dim_ordering("tf")


sio = socketio.Server()
app = Flask(__name__)

@sio.on('telemetry')
def telemetry(sid, data):
	# The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    img_str = data["image"]
    for f in filenames:
        steering_angle = (model.predict(f))
	throttle = .5
	print(steering_angle, throttle)
	send_control(steering_angle, throttle)
    # use a constant throttle
    #throttle = .5
    #print(steering_angle, throttle)
    #send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == "__main__":        
    filenames = glob.glob("dataset/round2/test/center/*.jpg")
    model = Model("dataset/models/weights_hsv_gray_diff_ch4_comma_prelu-final-00-0.10941.hdf5", "data/X_train_gray_diff2_mean.npy")
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
