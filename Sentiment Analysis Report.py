# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:54:12 2021

@author: Radu
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from stop_words import get_stop_words
from unidecode import unidecode

# Set path
path = 'C:/Users/Radu/Desktop/ML Projects/Bank FB Sentiment Analysis'
os.chdir(path)

data_comms=pd.read_csv(f'{path}/data_comms.csv')

# FB Follwers 31/05/2021
nr_follow = pd.DataFrame({'page': ['RaiffeisenBankRomania', 'BCR.Romania', 'INGWebCafe', 'Unicredit-343218583419079',  'BancaTransilvania',
                                  'OTPBankRomania', 'cecbank', 'BRDGroupeSocieteGenerale'],
                          'follower': [94902, 347505, 270868, 346, 603346, 50100, 18978, 117575]})


# Averge nr of likes per post
avg_likes = data_comms.groupby('page')['likes'].agg('mean').sort_values(ascending=False)

# Average posts per day
data_comms['post_time'] = pd.to_datetime(data_comms['post_time'], format = "%Y-%m-%d %H:%M:%S")
nr_days = ( data_comms.groupby('page')['post_time'].agg('max') - data_comms.groupby('page')['post_time'].agg('min')).astype('timedelta64[D]')
np_posts = data_comms.groupby('page')['post_id'].nunique()
avg_post_day = np_posts / nr_days
avg_post_day = avg_post_day.reset_index()

# Percent of negative/neuter/positive comments per post
data_comms['comm_label']  = np.select(
    [
        data_comms['prob_neg'].between(0, 0.4, inclusive=False), 
        data_comms['prob_neg'].between(0.4, 0.6, inclusive=True),
        data_comms['prob_neg'].between(0.6, 1, inclusive=False)
    ], 
    [
        'Positive', 
        'Neutral',
        'Negative'
    ], 
    default='Unknown'
)

pct_neg_comms = data_comms.groupby(['page','comm_label'])['comm_label'].agg('count')
pct_neg_comms = pct_neg_comms.rename('nr').reset_index()

# Raiffeisen negative comment word cloud
# importing all necessery modules
comment_words = ''
stopwords = get_stop_words('romanian') 
stopwords = stopwords + ['e', 'pt']

# iterate through the csv file
for val in data_comms[ (data_comms['comm_label'] == 'Positive')&(data_comms['page'] == 'RaiffeisenBankRomania') ].comment_text:
      
    # typecaste each val to string
    val = str(val)
  
    # split the value
    tokens = val.split()
    
    # Converts each token into lowercase & remove splacial accents (Romanian)
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        tokens[i] = unidecode(tokens[i])
        
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
  
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()

# Raiffeisen positive comment word cloud
# importing all necessery modules
comment_words = ''
stopwords = get_stop_words('romanian') 
stopwords = stopwords + ['e', 'pt']

# iterate through the csv file
for val in data_comms[ (data_comms['comm_label'] == 'Negative')&(data_comms['page'] == 'RaiffeisenBankRomania') ].comment_text:
      
    # typecaste each val to string
    val = str(val)
  
    # split the value
    tokens = val.split()
    
    # Converts each token into lowercase & remove splacial accents (Romanian)
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        tokens[i] = unidecode(tokens[i])
        
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
  
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
