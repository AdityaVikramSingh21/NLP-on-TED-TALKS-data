
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import nltk
import gensim
import re


# In[3]:


from nltk.corpus import PlaintextCorpusReader
corpus = PlaintextCorpusReader("C:/Users/Admin/Documents/Text_proj/Text_files (Removed Special Chars)", '.+\.txt',encoding='latin-1')
fids = corpus.fileids()


# In[4]:


docs = [corpus.words(f) for f in fids]


# In[5]:


docs2 = [[w.lower() for w in doc] for doc in docs]


# In[6]:


docs3 = [[w for w in doc if re.search('^[a-z]+$', w)] for doc in docs2]


# In[7]:


# Remove stop words.
from nltk.corpus import stopwords
stop_list = stopwords.words('english')
docs4 = [[w for w in doc if w not in stop_list] for doc in docs3]


# In[8]:


from nltk.stem.porter import *
stemmer = PorterStemmer()
docs5 = [[stemmer.stem(w) for w in doc] for doc in docs4]


# In[9]:


from gensim import corpora
dictionary = corpora.Dictionary(docs5)
print(dictionary)


# In[10]:


token_to_id = dictionary.token2id
print(type(token_to_id))


# In[11]:


print(token_to_id)


# In[12]:


vecs = [dictionary.doc2bow(doc) for doc in docs5]


# In[13]:


from gensim import similarities


# In[19]:


#sample_index = similarities.SparseMatrixSimilarity(vecs, 35398)


# In[ ]:


#for i in range(2414):
   # sims = sample_index[vecs[i]]
   # print(list(enumerate(sims)))


# In[14]:


from gensim import models
tfidf = models.TfidfModel(vecs)
vecs_with_tfidf = [tfidf[vec] for vec in vecs]


# In[15]:


# STEP 3 : Create similarity matrix of all files
index = similarities.MatrixSimilarity(vecs_with_tfidf)
sims = index[vecs_with_tfidf]
print(list(enumerate(sims)))


# In[31]:


top5=np.argsort(sims, axis=1)[:, -6:-1]


# In[32]:


top5


# In[30]:


sims[1, 1963]


# In[19]:


np.savetxt("top5.csv",top5, delimiter=",")

