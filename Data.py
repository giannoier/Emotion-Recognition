
import numpy as np
import os
import cv2
import itertools
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split

from keras.utils.np_utils import to_categorical







#////////////identifies txt file and returns its value as int///////////////////////////////////////////////////////////////////

def labeling_ck(emotion_path, folders, folders2):

    emotion = os.listdir(emotion_path + '/' + folders + '/' + folders2)

    for txt in emotion:
        text_file = open(emotion_path + '/' + folders + '/' + folders2 + '/' + txt, 'rb')
        text = int(float(text_file.readline()))
        text_file.close()
        return text


# //////////////////////////////////FACECROP_DETECT_FACE&CROP_IMAGE//////////////////////////////////////////////////////////////////


def facecrop(image):

    facedata = "haar-cascade/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [v for v in f]
        sub_face = img[y:y + h, x:x + w]

    return sub_face


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def DataLoad(test_path, data_path, emotion_path, folders_list,img_rows, img_cols):


    
    labels = []
    img_datalist = []
    img_testlist = []
    
    
#//////////////////////////////////////////////TEST///////////////////////////////////////////////////////////////////////////////////

    for img in os.listdir(test_path):
        test_data = cv2.imread(test_path + '/' + img)
        test_data = facecrop(test_path + '/' + img)
        test_data = cv2.cvtColor(test_data, cv2.COLOR_BGR2GRAY)
        test_image_resize = cv2.resize(test_data, (img_rows, img_cols))
        test_image_resize = cv2.equalizeHist(test_image_resize)
        img_testlist.append(test_image_resize)
    
    img_test = np.array(img_testlist)
    img_test = img_test.astype('float32')
    img_test /= 255
    
    img_test = np.expand_dims(img_test, axis=4) 

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    label2, img_data2 = fer2013()   # data coming from fer2013 dataset
    

    for folders in folders_list:
        folders_list2 = os.listdir(data_path + '/' + folders)
        print ('Loaded the images of dataset-' + '{}\n'.format(folders))
    
        for folders2 in folders_list2:
            folders_list3 = os.listdir(data_path + '/' + folders + '/' + folders2)
    
            for img in folders_list3:
                index = labeling_ck(emotion_path, folders, folders2)
                if (index is None):
                    break
                else:
                    label = np.zeros(7)
                    label[index - 1] = 1.0
                    labels.append(label)
                    img_data = cv2.imread(data_path + '/' + folders + '/' + folders2 + '/' + img)
                    img_data = facecrop(data_path + '/' + folders + '/' + folders2 + '/' + img)
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                    img_data = cv2.resize(img_data, (img_rows, img_cols))
                    img_data = cv2.equalizeHist(img_data)
                    #img_data = normalize(img_data, norm = 'l1')
                    img_datalist.append(img_data)
    
    
    
    img_data = np.array(img_datalist)
    img_data = img_data.astype('float32')
    img_data /= 255
    
    labels = np.array(labels)
    labels = np.concatenate((labels,label2))            #concatenate labels from fer and ck+
    np.set_printoptions(threshold = np.nan)
    print ("labels {}" .format(labels.shape))
    
    img_data = np.expand_dims(img_data, axis=4)
    
    img_data = np.concatenate((img_data,img_data2))     #concatenate data from fer and ck+
    
    
    num_samples = img_data.shape[0]
    X, y = shuffle(img_data, labels, random_state = 5)

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 4)

    return (X_train, Y_train, X_test, Y_test, img_test)


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