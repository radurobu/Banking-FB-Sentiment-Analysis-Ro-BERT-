# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:41:27 2021

@author: Radu
@article{dumitrescu2020birth,
  title={The birth of Romanian BERT},
  author={Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
  journal={arXiv preprint arXiv:2009.08712},
  year={2020}
}
@https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1
@https://github.com/dumitrescustefan/Romanian-Transformers
@https://medium.com/analytics-vidhya/hugging-face-transformers-how-to-use-pipelines-10775aa3db7e
@https://skok.ai/2020/05/11/Top-Down-Introduction-to-BERT.html
@https://github.com/pydatacluj/meetup-slides/blob/master/meetup_10/sentiment_analysis_pydata_cluj.ipynb
@Date comment-uri cu label: https://github.com/katakonst/sentiment-analysis-tensorflow/tree/master/datasets
Python Enviroment:
pip install urllib3==1.22
pip install transformers==3.5
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=9.2 -c pytorch
"""

#import sys
#sys.setrecursionlimit(10000)
import pandas as pd
import pathlib
from sklearn.datasets import load_files
import os
from transformers import AutoTokenizer, AutoModel
from torch import tensor, device, dtype, nn
import gc
from sklearn import model_selection

#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import optimizers
from keras import backend as K
from keras.models import model_from_json

# Set path
path = 'C:/Users/Radu/Desktop/ML Projects/Bank FB Sentiment Analysis'
os.chdir(path)

# import romanian BERT model (takes a couple of minutes)
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# tokenize a sentence and run through the model
tokens = tensor(tokenizer.encode("Acesta este un test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(tokens)
# get encoding
sequence_output = outputs[0]
pooled_output = outputs[1]

print(model)
print(tokens)
print(sequence_output)
print(pooled_output)

print(f'Length of BERT base vocabulary: {len(tokenizer.vocab)}')
for t in tokens[0]:
  print(f'Token: {t}, subword: {tokenizer.decode([t])}')
  
"""
The model outputs a tuple. The first item of the tuple has the following shape: 1 (batch size) x 7 (sequence length)
 x 768 (the number of hidden units). This is called the sequence output, and it provides the representation of each
 token in the context of other tokens in the sequence. If we'd like to fine-tune our model for named entity recognition,
 we will use this output and expect the 768 numbers representing each token in a sequence to inform us if the token
 corresponds to a named entity.

The second item in the tuple has the shape: 1 (batch size) x 768 (the number of hidden units).
 It is called the pooled output, and in theory it should represent the entire sequence.
 It corresponds to the first token in a sequence (the [CLS] token). We can use it in a text classification task
 - for example when we fine-tune the model for sentiment classification, we'd expect the 768 hidden units of the
 pooled output to capture the sentiment of the text.
 """
  
print(f'output type: {type(outputs)}, output length: {len(outputs)}')
print(f'first item shape: {outputs[0].shape}')
print(f'second item shape: {outputs[1].shape}')



# read comments data used for training sentiment analysis NN
def get_articles(source) -> pd.DataFrame:
    """
    data is taken from: # data is taken from: https://github.com/katakonst/sentiment-analysis-tensorflow/tree/master/datasets
    :return: the dataset containing labeled articles
    """
    # define paths where text files were placed
    path = pathlib.Path('C:/Users/Radu/Desktop/ML Projects/Bank FB Sentiment Analysis/Romanian Reviews/sentiment-analysis-tensorflow-master/datasets/ro/{}/'.format(source))
    
    # load files
    data = load_files(path, encoding="utf-8", decode_error="replace", random_state=500)
        
    # remove newlines
    data['data'] = [it.lower().replace('\n\n', ' ') for it in data['data']]

    # convert dict to Pandas dataframe
    df_raw = pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])
    
    # select only rows with non empty text
    df = df_raw[df_raw['text'] != '']
    
    return df

comms_train = get_articles(source="train")
comms_test = get_articles(source="test")
comms = pd.concat([comms_train,comms_test]).reset_index(drop=True)
comms.head(10)

# Split commnets into batches for training (otherwise we get "out of memory")
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks
comms_batch = split_dataframe(comms, chunk_size = 20)

# Apply ROBERT model on batches to get the pooled output
final_pooled_output = pd.DataFrame()
for batch in range(len(comms_batch)):
    print(f"We are at batch Nr: {batch}")
    sentences = comms_batch[batch].text.values
    tokens = tensor([tokenizer.encode(sent, add_special_tokens=True,max_length=128,pad_to_max_length=True) for sent in sentences])
    outputs = model(tokens)
    final_pooled_output = final_pooled_output.append(pd.DataFrame(outputs[1].detach().numpy())).reset_index(drop=True)
gc.collect()

# Save output and labels
final_pooled_output = pd.merge(final_pooled_output,comms['label'],left_index=True,right_index=True)
final_pooled_output.to_csv(f'{path}/final_pooled_output.csv',index=False)



# Train NN on sentiment output
final_pooled_output = pd.read_csv(f'{path}/final_pooled_output.csv')

X_train, X_test, y_train, y_test = model_selection.train_test_split(final_pooled_output.drop(columns=['label']), final_pooled_output['label'], test_size=0.2)

X_train = X_train.values
X_test = X_test.values

y_train = y_train.values
y_test = y_test.values

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 768, init = 'uniform', activation = 'relu', input_dim = 768, kernel_regularizer=regularizers.l2(0.001)))
# Adding dropout for regularization
classifier.add(Dropout(0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 576, init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
# Adding dropout for regularization
classifier.add(Dropout(0.1))

#Adding the third hidden layer
classifier.add(Dense(output_dim = 384, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
opt = optimizers.Adam(lr=0.001)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size = 5000, nb_epoch = 50)

from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.show()
pyplot.plot(history.history['val_loss'])
pyplot.show()

pyplot.plot(history.history['acc'])
pyplot.show()
pyplot.plot(history.history['val_acc'])
pyplot.show()

# Code inspired from: 
# - https://machinelearningmastery.com/save-load-keras-deep-learning-models/
def store_model(_model=model, path='./', file_prefix='model_sentiment_rnn'):
    
    # serialize model to JSON
    model_json = _model.to_json()
    json_file_path = os.path.join(path, "{}.json".format(file_prefix))
    print('Started saving model JSON to {}'.format(json_file_path))
    assert os.path.exists(path)
    with open(json_file_path, "w+") as json_file:
        json_file.write(model_json)
    print("Done saving model JSON to {}".format(json_file_path))
    
    # serialize weights to HDF5
    weights_file_path = os.path.join(path, '{}.h5'.format(file_prefix))
    print("Started saving model weights to {}".format(weights_file_path))
    _model.save_weights(weights_file_path)
    print("Done saving model weights to {}".format(weights_file_path))

def load_model(path='./', file_prefix='model_sentiment_rnn'):
    # load json and create model
    print('Started loading model from JSON file')
    with open('{}.json'.format(file_prefix), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    print('Done loading model from JSON file')
    
    # load weights into new model
    print("Starting loading model from disk")
    weights_file_path = os.path.join(path, '{}.h5'.format(file_prefix))
    assert os.path.exists(weights_file_path)
    loaded_model.load_weights(weights_file_path)
    print("Done loading model from disk from file: {}".format(weights_file_path))
    return loaded_model

# save model:
store_model(_model=classifier, path=path, file_prefix='model_sentiment')
#classifier = load_model(path=path, file_prefix='model_sentiment')

# Play with sentiment model

text='Toate bune o seară bună vouă oficial'

tokens = tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(tokens)
prob = classifier.predict_proba(outputs[1].detach().numpy())
print(f"Prbability of negative comment is: {1 - prob}")


text='Tot cu bucurie vă anunț că aplicația Smart mobile nu funcționează corect. La fel aplicația Raiffeisen on line prin browser. Mesaje nu putem da, la telefon aștept de peste 20 min să îmi păstrez prioritatea.'

tokens = tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(tokens)
prob = classifier.predict_proba(outputs[1].detach().numpy())
print(f"Prbability of negative comment is: {1 - prob}")

text='nu merge nimic'

tokens = tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(tokens)
prob = classifier.predict_proba(outputs[1].detach().numpy())
print(f"Prbability of negative comment is: {1 - prob}")
