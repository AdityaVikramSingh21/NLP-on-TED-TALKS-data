
# coding: utf-8

# In[42]:


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


# In[3]:


data = pd.read_csv('C:/Users/Admin/Documents/Text_proj/Worksheet_bins.csv', sep=',')


# In[4]:


data['transcript length'] = data['transcript'].apply(len)
data.head()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
g = sns.FacetGrid(data=data, col='Rating_I')
g.map(plt.hist, 'transcript length', bins=50)


# In[6]:


data_class = data[(data['Rating_I'] == 1) | (data['Rating_I'] == 4)]
data_class.shape


# In[7]:


X=data_class['transcript']
Y=data_class['Rating_I']


# In[8]:


import string
def preprocess(transcript):
    
    nopunc = [char for char in transcript if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[9]:


preprocess(X)


# In[12]:


np.random.seed(2018)
train = np.random.choice([True, False],len(X), replace=True, p=[0.5,0.5])
x_train = X[train]
y_train = Y[train]
x_test = X[~train]  # selecting test and train data
y_test = Y[~train]


# In[13]:


y_train.head()


# In[14]:


tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x_train = tfidf_vectorizer.fit_transform(x_train)
x_test = tfidf_vectorizer.transform(x_test)


# In[15]:


from sklearn.model_selection import cross_val_score
for i in [0.001,0.01,0.1,1,10]:
    mnb = naive_bayes.MultinomialNB(alpha = i)        #Multinomial Bayes with cross validation on test data
    mnb.fit(x_train, y_train)
    y_pred = mnb.predict(x_test)
    print('accuracy of Multinomial Naive Bayes: ', cross_val_score(mnb, x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(mnb, x_test, y_test)), np.std(cross_val_score(mnb, x_test, y_test))))


# In[17]:


mnb_best = naive_bayes.MultinomialNB(alpha = 0.01)
mnb_best.fit(x_train, y_train)                        #best multinomial Bayes model with alpha=0.1
y_pred = mnb_best.predict(x_test)
print('The best alpha is 0.01 and the test data accuracy is ', cross_val_score(mnb_best, x_test, y_test))
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(mnb_best, x_test, y_test)), np.std(cross_val_score(mnb_best, x_test, y_test))))


# In[18]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))


# In[19]:


feature_names = tfidf_vectorizer.get_feature_names()
_prob = np.zeros(len(feature_names))
_class = np.empty(len(feature_names), dtype='object')
for i in range(len(feature_names)):
    prob = mnb_best.feature_log_prob_[:, i]         #a dataframe that contains the feature_names,with the most probable category
    _prob[i] = np.amax(prob)
    _class[i] = 'Category %d' % np.argmax(prob == prob.max())
a = pd.DataFrame(np.stack((feature_names, _prob, _class)).T)


# In[20]:


mnb_best.classes_


# In[21]:


a1=a.loc[a[2] == 'Category 0']
a2=a.loc[a[2] == 'Category 1']


# In[24]:


a2.sort_values([1], ascending=False).head(15)  #to view the top ten words for each category 


# In[32]:


for i in [0.01,0.1,1,10,100,200,300]:
    logit = linear_model.LogisticRegression(C=i,multi_class='multinomial',solver='newton-cg')
    logit.fit(x_train, y_train)                     #multinomial logistic regression with cross validation on test data
    y_pred = logit.predict(x_test)                     
    print('accuracy of Logit Regression is: ', cross_val_score(logit, x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit, x_test, y_test)), np.std(cross_val_score(logit, x_test, y_test))))


# In[33]:


logit_best=linear_model.LogisticRegression(C=100,multi_class='multinomial',solver='newton-cg')
logit_best.fit(x_train,y_train)                 # best logistic regression model with C=1
y_pred = logit_best.predict(x_test)
print('The best alpha is 0.1 and the test data accuracy is ', cross_val_score(logit_best, x_test, y_test))
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit_best, x_test, y_test)), np.std(cross_val_score(logit_best, x_test, y_test))))


# In[26]:


rforest = ensemble.RandomForestClassifier(n_estimators=50,max_depth=20,random_state = 2018)
rforest.fit(x_train, y_train)
rforest.score(x_test, y_test)


# In[27]:


adaboost = [
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME', random_state = 2018),
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)]
for i in range(2):
    adaboost[i].fit(x_train, y_train)
    print(adaboost[i].score(x_test, y_test))


# In[28]:


parameters = [
    {'kernel': ['linear'], 'C':[0.1, 1, 10]},                      #with n_splits=7
    {'kernel': ['rbf'], 'gamma':[0.5, 1, 2], 'C':[0.1, 1, 10]}]

clf = model_selection.GridSearchCV(svm.SVC(), parameters,cv = model_selection.StratifiedKFold(n_splits = 7, shuffle = True, random_state = 2018))
clf.fit(x_train, y_train)
print('best score:', clf.best_score_)
print('best parameters: ', clf.best_params_)


# In[29]:


svm.SVC_best=svm.SVC(C=10,kernel='linear')
svm.SVC_best.fit(x_train,y_train)                       #best SVM on test data
print('test data accuracy of best Linear SVC is: ', cross_val_score(svm.SVC_best, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(svm.SVC_best, x_test, y_test)), np.std(cross_val_score(svm.SVC_best, x_test, y_test))))


# In[47]:


gboost = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 2018,max_depth = 20)
gboost.fit(x_train, y_train)
gboost.score(x_test, y_test)


# In[39]:


def get_fpr_tpr(clf, x_test, y_test):
    
    y_pred = pd.get_dummies(clf.predict(x_test))
    y_test_dum = pd.get_dummies(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(y_pred.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_dum.iloc[:, i], y_pred.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    print('tpr: ', tpr[0].shape[0], ', fpr: ', fpr[0].shape[0])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_dum.values.ravel(), y_pred.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_pred.shape[1])]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(y_pred.shape[1]):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= y_pred.shape[1]
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc, y_pred.shape[1]


# In[40]:


def plot_ROC_curve(fpr, tpr, auc_roc, num):
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(auc_roc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(auc_roc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    lw = 2
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for i in range(num):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.8f})'
                 ''.format(i, auc_roc[i]))
        
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# In[43]:


rf1 = rforest
rf1.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(rf1, x_test, y_test) #ROC curve for random forest
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[44]:


logit_best.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(logit_best, x_test, y_test) #ROC curve for best logistic regression
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[45]:


for i in range(2):
    adaboost[i].fit(x_train, y_train)
    fpr, tpr, auc_roc, num_class = get_fpr_tpr(adaboost[i], x_test, y_test) # ROC curve for ada boost 
    plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[46]:


svm.SVC_best.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(svm.SVC_best, x_test, y_test) # ROC curve for the best SVM
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[48]:


fpr, tpr, auc_roc, num_class = get_fpr_tpr(gboost, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)   # ROC curve for gradient boost


# In[49]:


fpr, tpr, auc_roc, num_class = get_fpr_tpr(mnb_best, x_test, y_test) #ROC curve for the best multinomial Bayes
plot_ROC_curve(fpr, tpr, auc_roc, num_class)  

