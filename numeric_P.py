
# coding: utf-8

# In[20]:


import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn import naive_bayes,linear_model,svm,model_selection,ensemble,tree,preprocessing
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp


# In[2]:


data = pd.read_csv('C:/Users/Admin/Documents/Text_proj/Work.csv', sep=',')


# In[3]:


data.head()


# In[4]:


data_class = data[(data['P_Rating'] == 1) | (data['P_Rating'] == 4)]
data_class.shape


# In[5]:


x=data_class.iloc[:,3:153]
y=data_class.iloc[:,-2]


# In[6]:


x.head()


# In[52]:


#x=x.drop(['posemo_change_h','negemo_change_h','affect_change_h','posemo_change_q','negemo_change_q','affect_change_q'],axis=1)


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


# In[8]:


x_train.head()


# In[9]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
dt = DecisionTreeClassifier(max_depth=10, random_state=4129)
#%time dt.fit(x_train, y_train)
get_ipython().run_line_magic('time', 'dt.fit(x_train, y_train == 4)')
pd.DataFrame(dt.feature_importances_, index=x_train.columns, columns=['imp']).sort_values(by='imp', axis=0, ascending=False).iloc[:16]


# In[159]:


''''mm_scale_train = preprocessing.MinMaxScaler().fit(x_train.iloc[:,0:144])
x_train_mm = mm_scale_train.transform(x_train.iloc[:,0:144])                # min-max Scalar transformation 
mm_scale_test = preprocessing.MinMaxScaler().fit(x_test.iloc[:,0:144])
x_test_mm = mm_scale_test.transform(x_test.iloc[:,0:144]) ''''


# In[173]:


''''std_scale_train = preprocessing.StandardScaler().fit(x_train.iloc[:,0:144])
x_train_std = std_scale_train.transform(x_train.iloc[:,0:144])
std_scale_test = preprocessing.StandardScaler().fit(x_test.iloc[:,0:144])  # standard Scalar transformation 
x_test_std = std_scale_test.transform(x_test.iloc[:,0:144])''''


# In[174]:


''''x_train_2= pd.DataFrame(x_train_std)
x_test_2= pd.DataFrame(x_test_std)''''


# In[175]:


''''x_train_2['posemo_change_h']=x_train.iloc[:,-6]
x_train_2['negemo_change_h']=x_train.iloc[:,-5]
x_train_2['affect_change_h']=x_train.iloc[:,-4]
x_train_2['posemo_change_q']=x_train.iloc[:,-3]
x_train_2['negemo_change_q']=x_train.iloc[:,-2]
x_train_2['affect_change_q']=x_train.iloc[:,-1]
x_test_2['posemo_change_h']=x_test.iloc[:,-6]
x_test_2['negemo_change_h']=x_test.iloc[:,-5]
x_test_2['affect_change_h']=x_test.iloc[:,-4]
x_test_2['posemo_change_q']=x_test.iloc[:,-3]
x_test_2['negemo_change_q']=x_test.iloc[:,-2]
x_test_2['affect_change_q']=x_test.iloc[:,-1]''''


# In[176]:


''''x_train_2.columns=x_train.columns
x_test_2.columns=x_test.columns''''


# In[ ]:





# In[72]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score


'''''def score_test(model, x_test):
    from pandas import DataFrame
    y_pred = model.predict(x_test)
    labels = ['4', '3', '2', '1']
    return DataFrame({
        'precision' : precision_score(y_test.P_Rating, y_pred, average=None, labels=labels),
        'recall' : recall_score(y_test.P_Rating, y_pred, average=None, labels=labels),
        'F1' : f1_score(y_test.P_Rating, y_pred, average=None, labels=labels),
        'kappa' : cohen_kappa_score(y_test.P_Rating, y_pred, labels=labels) },
        
        index=labels)''''''


# In[148]:





# In[10]:


from sklearn.model_selection import cross_val_score
for i in [0.001,0.01,0.1,1,10,100]:
    logit = linear_model.LogisticRegression(C=i,multi_class='multinomial',solver='newton-cg')
    logit.fit(x_train, y_train)                     #multinomial logistic regression with cross validation on test data
    y_pred = logit.predict(x_test)                     
    print('accuracy of Logit Regression is: ', cross_val_score(logit, x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit, x_test, y_test)), np.std(cross_val_score(logit, x_test, y_test))))


# In[11]:


logit_best=linear_model.LogisticRegression(C=1,multi_class='multinomial',solver='newton-cg')
logit_best.fit(x_train,y_train)                 # best logistic regression model with C=100
y_pred = logit_best.predict(x_test)
print('The best C is 1 and the test data accuracy is ', cross_val_score(logit_best, x_test, y_test))
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit_best, x_test, y_test)), np.std(cross_val_score(logit_best, x_test, y_test))))


# In[ ]:


'''''logit_best_data=pd.DataFrame(logit_best.coef_)
logit_best_data.columns=x.columns                     #getting top-ten words for each category 
logit_best_data.T.sort_values([3],ascending=False).head(10)''''''


# In[16]:


rforest = ensemble.RandomForestClassifier(n_estimators=50,max_depth=20,random_state = 2018)
rforest.fit(x_train, y_train)


# In[17]:


rforest.score(x_test, y_test)


# In[13]:


for i in [0.01,0.1,1, 10, 100]:
    linearSVM=svm.LinearSVC(C=i,multi_class='ovr')        #linear SVC with cross-validation on test data
    linearSVM.fit(x_train,y_train)
    print('accuracy of Linear SVC is: ', cross_val_score(linearSVM, x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(linearSVM, x_test, y_test)), np.std(cross_val_score(linearSVM, x_test, y_test))))


# In[14]:


parameters = [
    {'kernel': ['linear'], 'C':[0.1, 1, 10]},                      #with n_splits=7
    {'kernel': ['rbf'], 'gamma':[0.5, 1, 2], 'C':[0.1, 1, 10]}]

clf = model_selection.GridSearchCV(svm.SVC(), parameters,cv = model_selection.StratifiedKFold(n_splits = 7, shuffle = True, random_state = 2018))
clf.fit(x_train, y_train)
print('best score:', clf.best_score_)
print('best parameters: ', clf.best_params_)


# In[ ]:


#this model is doing a routine classification with numerics to predict persuasiveness the feture importance in this case will answer how we r saying it question.


# In[14]:


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


# In[15]:


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


# In[21]:


rf1 = rforest
rf1.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(rf1, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[22]:


logit_best.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(logit_best, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[23]:


adaboost = [
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME', random_state = 2018),
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)]
for i in range(2):
    adaboost[i].fit(x_train, y_train)
    print(adaboost[i].score(x_test, y_test))


# In[24]:


for i in range(2):
    adaboost[i].fit(x_train, y_train)
    fpr, tpr, auc_roc, num_class = get_fpr_tpr(adaboost[i], x_test, y_test)
    plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[26]:


best=svm.SVC(C=0.1,kernel='linear')
best.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(best, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[27]:


linearSVM=svm.LinearSVC(C=0.1,multi_class='ovr')
linearSVM.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(linearSVM, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[28]:


gboost = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 2018,max_depth = 20)
gboost.fit(x_train, y_train)
gboost.score(x_test, y_test)


# In[29]:


fpr, tpr, auc_roc, num_class = get_fpr_tpr(gboost, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)

