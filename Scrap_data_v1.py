# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:07:48 2021

@author: Radu

@article{dumitrescu2020birth,
  title={The birth of Romanian BERT},
  author={Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
  journal={arXiv preprint arXiv:2009.08712},
  year={2020}
}

@https://pypi.org/project/facebook-scraper/
"""

#https://www.facebook.com/RaiffeisenBankRomania
#https://www.facebook.com/BCR.Romania
#https://www.facebook.com/INGWebCafe
#https://www.facebook.com/Unicredit-343218583419079 (au renuntat la FB si Instagram din JUN2019)
#https://www.facebook.com/BancaTransilvania/
#https://www.facebook.com/OTPBankRomania/
#https://www.facebook.com/cecbank/
#https://www.facebook.com/BRDGroupeSocieteGenerale
from facebook_scraper import get_posts
import time
import random
import json
import os
path = 'C:/Users/Radu/Desktop/ML Projects/Bank FB Sentiment Analysis'
os.chdir(path)

FB_pages = ['RaiffeisenBankRomania', 'BCR.Romania', 'INGWebCafe', 'Unicredit-343218583419079',  'BancaTransilvania',
                'OTPBankRomania', 'cecbank', 'BRDGroupeSocieteGenerale']

#FB_pages = ['OTPBankRomania', 'cecbank', 'BRDGroupeSocieteGenerale']

for page_name in FB_pages:
    
    print(f"We are at page: {page_name}")
    full_data={}
    for nr, post in enumerate(get_posts(page_name, pages=25, extra_info=True, options={"comments": True,"reactors": True})):
        print(nr)
        full_data[nr] = post
        time.sleep(random.randint(2, 11))
        
    with open(f'{page_name}_Posts.json', 'w') as fp:
        json.dump(full_data, fp, indent=4, sort_keys=False, default=str)
    time.sleep(random.randint(60, 180))
    
    
