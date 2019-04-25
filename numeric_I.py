
# coding: utf-8

# In[22]:


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


# In[23]:


data = pd.read_csv('C:/Users/Admin/Documents/Text_proj/Work.csv', sep=',')


# In[24]:


data.head()


# In[25]:


data_class = data[(data['I_Rating'] == 1) | (data['I_Rating'] == 4)]
data_class.shape


# In[26]:


x=data_class.iloc[:,3:153]
y=data_class.iloc[:,-1]


# In[27]:


x.head()


# In[28]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


# In[29]:


x_train.head()


# In[30]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
dt = DecisionTreeClassifier(max_depth=10, random_state=4129)
get_ipython().run_line_magic('time', 'dt.fit(x_train, y_train)')
get_ipython().run_line_magic('time', 'dt.fit(x_train, y_train == 4)')
pd.DataFrame(dt.feature_importances_, index=x_train.columns, columns=['imp']).sort_values(by='imp', axis=0, ascending=False).iloc[:16]


# In[31]:


from sklearn.model_selection import cross_val_score
for i in [0.001,0.01,0.1,1,10,100]:
    logit = linear_model.LogisticRegression(C=i,multi_class='multinomial',solver='newton-cg')
    logit.fit(x_train, y_train)                     #multinomial logistic regression with cross validation on test data
    y_pred = logit.predict(x_test)                     
    print('accuracy of Logit Regression is: ', cross_val_score(logit, x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit, x_test, y_test)), np.std(cross_val_score(logit, x_test, y_test))))


# In[32]:


logit_best=linear_model.LogisticRegression(C=0.1,multi_class='multinomial',solver='newton-cg')
logit_best.fit(x_train,y_train)                 # best logistic regression model with C=100
y_pred = logit_best.predict(x_test)
print('The best C is 0.1 and the test data accuracy is ', cross_val_score(logit_best, x_test, y_test))
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit_best, x_test, y_test)), np.std(cross_val_score(logit_best, x_test, y_test))))


# In[33]:


rforest = ensemble.RandomForestClassifier(n_estimators=50,max_depth=20,random_state = 2018)
rforest.fit(x_train, y_train)
rforest.score(x_test, y_test)


# In[34]:


for i in [0.01,0.1,1, 10, 100]:
    linearSVM=svm.LinearSVC(C=i,multi_class='ovr')        #linear SVC with cross-validation on test data
    linearSVM.fit(x_train,y_train)
    print('accuracy of Linear SVC is: ', cross_val_score(linearSVM, x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(linearSVM, x_test, y_test)), np.std(cross_val_score(linearSVM, x_test, y_test))))


# In[35]:


parameters = [
    {'kernel': ['linear'], 'C':[0.1, 1, 10]},                      #with n_splits=7
    {'kernel': ['rbf'], 'gamma':[0.5, 1, 2], 'C':[0.1, 1, 10]}]

clf = model_selection.GridSearchCV(svm.SVC(), parameters,cv = model_selection.StratifiedKFold(n_splits = 7, shuffle = True, random_state = 2018))
clf.fit(x_train, y_train)
print('best score:', clf.best_score_)
print('best parameters: ', clf.best_params_)


# In[36]:


adaboost = [
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME', random_state = 2018),
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)]
for i in range(2):
    adaboost[i].fit(x_train, y_train)
    print(adaboost[i].score(x_test, y_test))


# In[37]:


gboost = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 2018,max_depth = 20)
gboost.fit(x_train, y_train)
gboost.score(x_test, y_test)


# In[38]:


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


# In[39]:


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


# In[40]:


logit_best.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(logit_best, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[41]:


rforest.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(rforest, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[42]:


best=svm.LinearSVC(C=0.1,multi_class='ovr')
best.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(best, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[45]:


best_SVC=svm.SVC(C=0.1,kernel='linear')
best_SVC.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(best_SVC, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[46]:


for i in range(2):
    adaboost[i].fit(x_train, y_train)
    fpr, tpr, auc_roc, num_class = get_fpr_tpr(adaboost[i], x_test, y_test)
    plot_ROC_curve(fpr, tpr, auc_roc, num_class)


# In[47]:


gboost.fit(x_train, y_train)
fpr, tpr, auc_roc, num_class = get_fpr_tpr(gboost, x_test, y_test)
plot_ROC_curve(fpr, tpr, auc_roc, num_class)

