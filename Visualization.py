


import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import itertools




#////////////////////////////////////////confusion_matrix////////////////////////////////////////////////////////////////////////////


def plot_confusion_matrix(cm,
                          normalize,
                          classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
    plt.show()

    #////////////////////////////////////////////////Feature_Map_Visualization/////////////////////////////////////////////////////////////

def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations


def VisualizeFeatureMaps(model, img_test):

    layer_num=2
    filter_num=0
    
    activations = get_featuremaps(model, int(layer_num),img_test)
    
    print (np.shape(activations))
    feature_maps = activations[0][0]      
    print (np.shape(feature_maps))
    
    fig=plt.figure(figsize=(12,12))
    plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
    plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.png')
    
    num_of_featuremaps=feature_maps.shape[2]
    fig=plt.figure(figsize=(12,12)) 
    plt.title("featuremaps-layer-{}".format(layer_num))
    subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
        ax = fig.add_subplot(subplot_num, subplot_num, i+1)
        #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
        ax.imshow(feature_maps[:,:,i],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("featuremaps-layer-{}".format(layer_num) + '.png')
    

