
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
#from keras.preprocessing import image as kimage
from random import randint,shuffle
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import to_categorical

# In[1]:

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Concatenate, Dropout, Lambda, Multiply, RepeatVector, Permute
from keras.layers import Input
from keras.callbacks import EarlyStopping
import keras.backend as K
import sys
import os as os
import codecs


# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

# if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "2"



# In[32]:

vocab = list(np.load('vocab.npy'))
image_size, sent_size = list(np.load('parameters.npy'))
#labels = ['POS', 'NEU', 'NEG']


# In[4]:

#angry_images=np.load('angry_images.npy')
#happy_images=np.load('happy_images.npy')
angry_sents=np.load('angry_sents.npy')
happy_sents=np.load('happy_sents.npy')


# In[20]:

all_data=[]

for sent in happy_sents:
    all_data.append((sent[0],sent[1]))
    
for sent in angry_sents:
    all_data.append((sent[0],sent[1]))


# In[21]:

len(all_data)


# In[24]:

shuffle(all_data)


# In[25]:

s=int((len(all_data)*0.8))#splits with 80% training, 20% test
train, test=all_data[:s], all_data[s:]


# In[26]:

xtrain = [sent for (sent,score) in train]
ytrain1 =[score for (sent,score) in train ]
#ytrain1=[0 for (_,score) in train if score=='NEG']
xtest =[sent for (sent,score) in test]
ytest1 =[score for (sent,score) in test]
#ytest1=[1 for (_,score) in test if score=='POS']
#ytest1=[0 for (_,score) in test if score=='NEG']
ytrainpos = [1 for i in ytrain1 if i =='POS']
ytrainneg = [0 for i in ytrain1 if i =='NEG']

ytrain =  ytrainpos + ytrainneg
#ytrain = to_categorical(ytrainpos + ytrainneg)
#xtrain = [i[0] for i in xtrain1 if i[1] =='POS' or i[1]=='NEG']
#ytrain = [i[1] for i in ytrain1 if i[1] =='POS' or i[1]=='NEG']

ytestpos = [1 for i in ytest1 if i =='POS']
ytestneg = [0 for i in ytest1 if i =='NEG']

ytest =  ytestpos + ytestneg

np.save("xtrain_language.npy", xtrain)
np.save("ytrain_language.npy", ytrain)
np.save("xtest_language.npy", xtest)
np.save("ytest_language.npy", ytest)

# In[29]:
print(len(xtest))
print(len(ytest))


sent_size=len(all_data[0][0])

# Adaption of Mehdi’s code for Visual QA:
# 
# input question is the sentence --- input image is image --- output answer is the sentiment label

# In[36]:

input_question = Input([sent_size,])
#input_context = Input([image_size,])

# padding token is 0, other values are valid:
q_mask_0 = Lambda(lambda x: K.clip(x, 0, 1))(input_question)
q_embs_0 = Embedding(len(vocab), 50)(input_question)

q_mask = Permute((2,1))(RepeatVector(50)(q_mask_0))

q_embs = Multiply()([q_mask, q_embs_0])
# encode the question
q_encoded = Dropout(0)(LSTM(50)(q_embs))

mlp_1 = Dense(1, activation='relu')(q_encoded)
#mlp_1 = Dense(1, activation='softmax')(q_encoded)


#mlp_2 = Dropout(0)(Dense(image_size, activation='relu')(mlp_1))

#final_a = Dense(1, activation='relu')(mlp_1)

# model 1: language only sentiment
model1 = Model(input_question, mlp_1)
model1.summary()


# model 2: language+face sentiment
#q_composed = Concatenate()([input_context, mlp_1])
#mlp_3 = Dropout(0)(Dense(image_size, activation='relu')(q_composed))
#final_b = Dense(len(labels), activation='softmax')(mlp_3)

#model2 = Model([input_question, input_context], final_b)
#model2.summary()


# In[34]:


#model1.compile('adam', 'mean_squared_error', ['mean_squared_error', 'accuracy'])
model1.compile('adam', 'binary_crossentropy', ['accuracy'])


#working
model1.fit(xtrain, ytrain, epochs=100, batch_size=128, validation_split=0.1, callbacks=[EarlyStopping(patience=50)])



model1.save('language_model.h5')  # creates a HDF5 file 'my_model.h5'



scores = model1.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



# model = load_model('saved_model.h5')




