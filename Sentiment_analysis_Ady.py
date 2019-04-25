
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics,naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import re
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, feature_extraction, naive_bayes, linear_model, svm,decomposition,model_selection,tree,ensemble
from nltk.corpus import stopwords
import nltk
from itertools import cycle
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp


# In[4]:


data = pd.read_csv('for_sentiment.csv', sep=',')


# In[5]:


data.head()


# In[6]:


X=data['transcript']


# In[7]:


len(X)


# In[8]:


from  nltk.sentiment.vader  import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


# In[9]:


def analyze_sentiment_vader_lexicon(transcript, 
                                    threshold=0.1,
                                    verbose=False):
    
    
    # analyze the sentiment for talk
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(transcript)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold                                   else 'negative'
    if verbose:
        # display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative', 'Neutral']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print(sentiment_frame)
    
    return final_sentiment


# In[10]:


for transcript in (X):
    pred = analyze_sentiment_vader_lexicon(transcript, threshold=0.4, verbose=True)    

