from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from random import randint,shuffle

d=open("dictionary.txt", 'r', encoding='utf-8')
s=open("sentiment_labels.txt", encoding='utf-8')
sents=[]
labels=[]
for line in d:
    sents.append(line.split('|')[0])
for line in s:
    score=line.split('|')[1]
    labels.append(score.split('\n')[0])
d.close()
s.close()

print('sents', len(sents))
def pad_vector(size, sent_feat_temps):
    return [0]*(size-len(sent_feat_temps)) + sent_feat_temps
def create_vocab(fts_list):#takes a list of lists of words
    rV=[]
    for el in fts_list:
        for word in el:
            rV.extend(el)
    rV=set(rV)
    return ['<pad>'] + list(rV)


token_sents=[]
lemmatizer = WordNetLemmatizer()
for sent in sents:
    sentence=word_tokenize(sent)
    lem_sentence=[]
    for word in sentence:
        lem_sentence.append(lemmatizer.lemmatize(word.lower()))
    token_sents.append(lem_sentence)

#    print(sentence)
#    input()
zipped=list(zip(token_sents,labels))
del zipped[0]#getting rid of the header
zipped.sort(key = lambda x: len(x[0]), reverse=True)
language_data=zipped[:150000]
#list of tuples. One per sentence. ([list of lemmatized lowercase words in the sentence], a sentiment score)

sentlist=[sent for sent,score in language_data]
longest_sent = max(len(sent) for sent in sentlist)
vocab=create_vocab(sentlist)
vocabsize=len(vocab)
inttofeat = dict(zip(range(vocabsize), vocab))
feattoint = dict(zip(vocab, range(vocabsize)))

print('language_data', len(language_data))

padded_data=[]
for sent,score in language_data:
    feat=[feattoint[word] for word in sent]
    padded_data.append((pad_vector(longest_sent, feat),score))

print('padded_data', len(padded_data))

#Now, sort the sentences in 2 categories:

angry_sents=[]
happy_sents=[]

np.save('angry_sents.npy', angry_sents)
np.save('happy_sents.npy', happy_sents)

for sent,score in padded_data:
    if 0.0 <= float(score) <= 0.40:
        angry_sents.append((sent,score))
#    elif float(score) == 0.50:
#        neutral_sents.append((sent,score))
    elif 0.60 <= float(score) <= 1:
        happy_sents.append((sent,score))
        
print('angry', len(angry_sents))
print('happy', len(happy_sents))
#print('neutral', len(neutral_sents))

df=pd.read_csv("fer2013/fer2013.csv")#image data

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
angry_df=df[df.emotion == 0]
happy_df=df[df.emotion == 3]
#neutral_df=df[df.emotion == 6]


#######################################################################
#save to folder as png-files
#count=0
#for row in angry_df.iterrows():
#    image_feat=angry_df.iloc[count]['pixels'].split()
#    image_feat = [int(i) for i in image_feat]
#    new=np.array(image_feat).reshape((48,48)).astype(np.uint8)
#    im = Image.fromarray(new)
#    im.save('pngs/angry/output{}.png'.format(count))
#    count=count+1
    
#count=0
#for row in happy_df.iterrows():
#    image_feat=happy_df.iloc[count]['pixels'].split()
#    image_feat = [int(i) for i in image_feat]
#    new=np.array(image_feat).reshape((48,48)).astype(np.uint8)
#    im = Image.fromarray(new)
#    im.save('pngs/happy/output{}.png'.format(count))
#    count=count+1

########################################################################

#print('Negative sentences:  ', len(angry_sents))

#print('Positive sentences:  ', len(happy_sents))

all_data=[]
for sent,score in angry_sents:#[:7500]:
    image_feat=angry_df.iloc[randint(0, len(angry_df)-1)]['pixels'].split()
    image_feat = [int(i) for i in image_feat]
    all_data.append((np.array(image_feat),sent,'NEG'))
                    
for sent,score in happy_sents:#[:7500]:
    image_feat=happy_df.iloc[randint(0, len(happy_df)-1)]['pixels'].split()
    image_feat = [int(i) for i in image_feat]
    all_data.append((np.array(image_feat),sent,'POS'))

#for sent,score in neutral_sents:
#    image_feat=neutral_df.iloc[randint(0, len(neutral_df)-1)]['pixels'].split()
#    image_feat = [int(i) for i in image_feat]
#    all_data.append((np.array(image_feat),sent,'NEU'))
print('data points: ',len(all_data))

shuffle(all_data)

s=int((len(all_data)*0.8))#splits with 80% training, 20% test
train, test=all_data[:s], all_data[s:]

xtrain=[(image,sent) for (image,sent,label) in train]
ytrain=[label for (image,sent,label) in train]
xtest=[(image,sent) for (image,sent,label) in test]
ytest=[label for (image,sent,label) in test]

image_size=len(all_data[0][0])
sent_size=len(all_data[0][1])
labels=set(list(l for (i,s,l) in all_data))

xtrain2 = list(zip(*xtrain))
xtrain2 = [np.array(xtrain2[1]), np.array(xtrain2[0])]

xtest2 = list(zip(*xtest))
xtest2 = [np.array(xtest2[1]), np.array(xtest2[0])]


cat_codes = {
    'POS': 0,
    'NEU': 1,
    'NEG': 2,
}
ytrain2 = np.array([cat_codes[c] for c in ytrain])
ytest2 = np.array([cat_codes[c] for c in ytest])

np.save('parameters.npy',[image_size, sent_size]) 
np.save("vocab.npy", vocab)
np.save("xtrain0.npy", xtrain2[0])
np.save("xtrain1.npy", xtrain2[1])
np.save("ytrain.npy", ytrain2)
np.save("xtest0.npy", xtest2[0])
np.save("xtest1.npy", xtest2[1])
np.save("ytest.npy", ytest2)
