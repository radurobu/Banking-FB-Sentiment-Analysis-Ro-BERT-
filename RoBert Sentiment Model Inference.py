# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:19:38 2021

@author: Radu
"""
# Libraries
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
from torch import tensor, device, dtype, nn
import gc
import json
from keras.models import model_from_json

# Set path
path = 'C:/Users/Radu/Desktop/ML Projects/Bank FB Sentiment Analysis'
os.chdir(path)

# Usefull functions
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

# Embeding Model
# import romanian BERT model (takes a couple of minutes)
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

# Classifier Model
classifier = load_model(path=path, file_prefix='model_sentiment')

# Prepare scrapped data for model infference
FB_pages = ['RaiffeisenBankRomania', 'BCR.Romania', 'INGWebCafe', 'Unicredit-343218583419079',  'BancaTransilvania',
                'OTPBankRomania', 'cecbank', 'BRDGroupeSocieteGenerale']
    
data_comms = pd.DataFrame()  
for name in FB_pages:
    print(name)
    with open(f"{name}_Posts.json", "r") as read_file:
        X = json.load(read_file)
        for i in X.keys():
            range_comms = X[f'{i}']['comments_full']
            if range_comms is None: #Check if no comments in post
                pass
            else:
                for j in range(len(range_comms)):
                    data_comms = data_comms.append(pd.DataFrame({'page': name,
                                                                 'comment_id': [ X[f'{i}']['comments_full'][j]['comment_id'] ],
                                                                 'post_id': [ X[f'{i}']['post_id'] ],
                                                                 'post_time': [ X[f'{i}']['time'] ],
                                                                 'likes': [ X[f'{i}']['likes'] ],
                                                                 'nr_comments': [ X[f'{i}']['comments'] ],
                                                                 'comment_text': [ X[f'{i}']['comments_full'][j]['comment_text'] ],
                                                                 'comment_time': [ X[f'{i}']['comments_full'][j]['comment_time'] ]
                                                                                                         })).reset_index(drop=True)

with open(f"OTPBankRomania_Posts.json", "r") as read_file:
    X = json.load(read_file)
    for i in X.keys():
        range_comms = X[f'{}']['comments_full']

del X
              
# Apply Model:
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks
comms_batch = split_dataframe(data_comms, chunk_size = 20)

# Apply ROBERT model on batches to get the pooled output
final_pooled_output = pd.DataFrame()
for batch in range(len(comms_batch)):
    print(f"We are at batch Nr: {batch}")
    sentences = comms_batch[batch].comment_text.values
    tokens = tensor([tokenizer.encode(sent, add_special_tokens=True,max_length=128,pad_to_max_length=True) for sent in sentences])
    outputs = model(tokens)
    final_pooled_output = final_pooled_output.append(pd.DataFrame(outputs[1].detach().numpy())).reset_index(drop=True)
gc.collect()

# Apply clasiffier to ROBERT outputs
prob = classifier.predict_proba(final_pooled_output)

# Final prediction
data_comms["prob_neg"] = 1 - prob
data_comms.to_csv(f'{path}/data_comms.csv',index=False)
means = data_comms.groupby(data_comms["page"])["prob_neg"].mean()
print(means)
