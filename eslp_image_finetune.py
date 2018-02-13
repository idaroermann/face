'''This script goes along the blog post

"Building powerful image classification models using very little data"

from blog.keras.io.

It uses data that can be downloaded at:

https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:

- created a data/ folder

- created train/ and validation/ subfolders inside data/

- created cats/ and dogs/ subfolders inside train/ and validation/

- put the cat pictures index 0-999 in data/train/cats

- put the cat pictures index 1000-1400 in data/validation/cats

- put the dogs pictures index 12500-13499 in data/train/dogs

- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.

In summary, this is our directory structure:

```

data/

    train/

        dogs/

            dog001.jpg

            dog002.jpg

            ...

        cats/

            cat001.jpg

            cat002.jpg

            ...

    validation/

        dogs/

            dog001.jpg

            dog002.jpg

            ...

        cats/

            cat001.jpg

            cat002.jpg

            ...

```

'''



from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense

import matplotlib.pyplot as plt



import os

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]= "3"

import h5py





# path to the model weights files.

weights_path = 'vgg16_weights.h5'

top_model_weights_path = 'bottleneck_fc_model.h5'
#top_model_weights_path = h5py.File('bottleneck_fc_model.h5', mode='r')



# dimensions of our images.

img_width, img_height = 48, 48#150, 150



train_data_dir = 'data/train'

validation_data_dir = 'data/test'

nb_train_samples = 7904#7930

nb_validation_samples = 1952#1980

epochs = 50

batch_size = 64



# build the VGG16 network

model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width,img_height,3))# I added , input_shape = (48,48,3)

print('Model loaded.')



# build a classifier model to put on top of the convolutional model

top_model = Sequential()

top_model.add(Flatten(input_shape=model.output_shape[1:]))

top_model.add(Dense(256, activation='relu'))

top_model.add(Dropout(0.5))

top_model.add(Dense(1, activation='sigmoid'))



# note that it is necessary to start with a fully-trained

# classifier, including the top classifier,

# in order to successfully do fine-tuning

top_model.load_weights(top_model_weights_path)



# add the model on top of the convolutional base

#model.add(top_model)###I commented this out and made a workaround below:







###################################I needed a work-around to add the top model: ^ ####################################################

new_model = Sequential()######I added this part

for l in model.layers:######

    new_model.add(l)#######



# set the first 25 layers (up to the last conv block)

# to non-trainable (weights will not be updated)

#for layer in model.layers[:25]:######## and I commented this out

#    layer.trainable = False############ and I commented this out



#Concatenate the two models

new_model.add(top_model)



# LOCK THE TOP CONV LAYERS

for layer in new_model.layers:

    layer.trainable = False



#######################################################################################



# compile the model with a SGD/momentum optimizer

# and a very slow learning rate.

new_model.compile(loss='binary_crossentropy',#####################added new_ in the beginning from my workaround ^

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



# prepare data augmentation configuration

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary')



# fine-tune the model

##(The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. 

#`steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. 

#Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. 

#Update your method calls accordingly:)



# new_model.fit_generator(

#     train_generator,

#     samples_per_epoch=nb_train_samples,

#     epochs=epochs,

#     validation_data=validation_generator,

#     nb_val_samples=nb_validation_samples)

#Updated metod calls:

history = new_model.fit_generator(

train_generator,

steps_per_epoch=nb_train_samples//batch_size,

epochs=epochs,

validation_data=validation_generator,

nb_val_samples=nb_validation_samples//batch_size)







#my adding:

new_model.save('image_model.h5')

print(history.history)

modelfile = open('./imagehistory.txt',"w")

modelfile.write(str(history.history))

modelfile.close()

### summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig('imageloss.pdf')

### summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig('imageaccuracy.pdf')