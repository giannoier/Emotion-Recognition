from Visualization import VisualizeFeatureMaps, plot_confusion_matrix
from Model import Gen_model
from Data import DataLoad
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix



import numpy as np
import os


from keras import backend as K

DATA_AUGMENTATION = True 
PATH = os.getcwd()
test_path = PATH + '/test'
data_path = PATH + '/data'
emotion_path = PATH + '/emotion'
folders_list = os.listdir(data_path)

img_rows, img_cols = 48, 48
# batch size to train
batch_size = 128
# number of output classes
nb_classes = 7
# number of epochs to train
epochs = 100
# number of channels
img_channels = 1
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolutional kernel size
nb_conv = 3


Classes = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

data_generator = ImageDataGenerator(
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=.1,
                horizontal_flip=False)


X_train, Y_train, X_test, Y_test, img_test = DataLoad(test_path, data_path, emotion_path, folders_list, img_rows, img_cols)

print ("Xtrain shape: --{}" .format(X_train.shape))

model = Gen_model(img_rows, img_cols)
filename = PATH + '/modelcsv' + '/model_train_new.csv'
csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

filepath = PATH + '/hd5' + '/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hd5'
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log, checkpoint]
if DATA_AUGMENTATION:
    history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch = len(X_train) / batch_size, 
        epochs = epochs, verbose=1, 
        validation_data=(X_test, Y_test), callbacks = callbacks_list)
else:
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs = epochs, verbose=1, validation_data=(X_test, Y_test), callbacks = callbacks_list)




if __name__ == "__main__":


    score = model.evaluate(X_test, Y_test, verbose=0)
        
    print ("Validaton Loss", score[0])
    print ("Validation accuracy", score[1])
    
    y_pred = (model.predict_classes(X_test))
    print ("ypredd {}" .format(y_pred))
    
    y_true = np.argmax(Y_test,1)        #Convert OneHotEncoding to RealValue Number with argmax function
    
    

    cnf_matrix = confusion_matrix(y_true, y_pred)           
    
    #VisualizeFeatureMaps(model, img_test) 
    plot_confusion_matrix(cnf_matrix, classes = Classes, title = 'plot_confusion_matrix', normalize = True)
    

    np.set_printoptions(precision=2)
    print (precision_recall_fscore_support(y_true, y_pred, average = 'macro'))

    






