
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers




def Gen_model(img_rows, img_cols):


    input_shape = img_rows, img_cols, 1
    

    model = Sequential()
    
    #1stCONV2d
    model.add(Conv2D(32, (5, 5), padding ='same', input_shape = input_shape))
    model.add(Activation('relu'))
    
    #2ndCONV2d
    model.add(ZeroPadding2D((1,1),input_shape= input_shape))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    #3rdCONV2d
    model.add(ZeroPadding2D((1,1),input_shape= input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #4thCONV2D
    model.add(ZeroPadding2D((1,1),input_shape= input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #5thCONV2D
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    #fully connected NN
    model.add(Dense(1024))
    model.add(Activation('relu'))   
    model.add(Dropout(0.3))

    model.add(Dense(612))
    model.add(Activation('relu'))   
    model.add(Dropout(0.2))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    adadelta = optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])

    model.summary()
    
    

    return(model)
