# face
ESLP project
Lerning sentiment scores from images through sentences. 


fer2013 is the image data. It is too big to upload on GitHub as well (max 25 mb). The link is here https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
This data is changed to png files in folders. See code in preprocessing.py. 


1. Language only:

dictionary.txt and sentiment_labels.txt is the language data. ( https://nlp.stanford.edu/sentiment/code.html Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts, Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, Conference on Empirical Methods in Natural Language Processing (EMNLP 2013))

preprocessing.py takes the language data
—> outputs vocab.npy, parameters.npy, happy_sents and angry_sents.npy

elsp_language.py takes this info
—> outputs language_model.h5


2. Image only:

followed this blog https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html 
and fixed the image data from .csv to .png in a path like data/
train/
happy/
happy0.png

, accordingly for test and angry images. 
I balanced out the data: 3965/990 train/test happy, 3965/990 train/test angry

The script is elsp_images.py and takes only the data-path as input.
—> outputs bottleneck_fc_model.h5 , that uses the VGG16 network pre-trained on ImageNet. 

I had to fix some stuff to make the code running (numpy changes and keras 2)

This model is fine-tuned in elsp_image_finetune.py , that fits and save image_model.h5 in your directory.


3. Vision + language: 

The image_model.h5 is loaded in combi.py. This script takes the happy_sents and angry_sents, and the path to the pngs (that is ordered in png/happy/.. and png/angry/..  . 




