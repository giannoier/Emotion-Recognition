
import pandas as pd
import numpy as np
from random import shuffle
import cv2
from keras.utils.np_utils import to_categorical


def fer2013():

    
    # adjust fer data labels to ck+ (emotion_labels(1-5) become (2-6) zero remains )
    data = pd.read_csv('fer2013.csv')
    
    anger = data[(data.emotion == 0)]
    anger_emotion = anger['emotion'].tolist()

    anger_pixels = anger['pixels'].tolist()

    data = data[(((data.emotion) < 6) & ((data.emotion) != 0))]
    
    pixels = data['pixels'].tolist()
    pixels = pixels + anger_pixels
    emotions = data['emotion'].tolist()

    emotions = list(map(lambda x:x+1, emotions))           

    emotions = emotions + anger_emotion
    emotions = to_categorical(emotions,7)
    
    width, height = 48, 48
    faces = []
    
     
    
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48,48))
        face = cv2.equalizeHist(face)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces /= 255
    faces = np.expand_dims(faces, -1)
    
    
    return(emotions,faces)



