
# coding: utf-8

# # Problem Statement:

# ## Problem 1: Prediction task is to determine whether a person makes over 50K a year.

# ### The description of the dataset is as follows:

# <p><b>Data Set Information:</b></p>
# <p>Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))</p>
# <p><b>Attribute Information:</b></p>
# <p>Listing of attributes: &lt;50K, &gt;=50K.</p>
# <p>age: continuous. </p>
# <p>workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.</p>
# <p>fnlwgt: continuous.</p>
# <p>education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc,9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.</p>
# <p>education-num: continuous.</p>
# <p>marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.</p>
# <p>occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial,Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing,</p>
# <p>Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.</p>
# <p>relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.</p>
# <p>race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.</p>
# <p>sex: Female, Male.</p>
# <p>capital-gain: continuous.</p>
# <p>capital-loss: continuous.</p>
# <p>hours-per-week: continuous.</p>
# <p>native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands</p>

# ## Importing Modules 

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from xgboost import plot_tree
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## Loading Data

# In[2]:


train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# ## Data Exploration

# In[3]:


train_set.head() #Returns the first 5 rows of train_set dataframe


# In[4]:


test_set.head() #Returns the first 5 rows of test_set dataframe


# In[5]:


train_set.info() #Prints information about train_set dataframe.


# In[6]:


test_set.info() #Prints information about test_set dataframe.


# In[7]:


train_set.describe() #The summary statistics of the train_set dataframe


# In[8]:


test_set.describe() #The summary statistics of the test_set dataframe


# In[9]:


train_set.isnull().values.any() #Check for any NA’s in the dataframe.


# In[10]:


test_set.isnull().values.any() #Check for any NA’s in the dataframe.


# In[11]:


train_set['workclass'].value_counts() #Returns object containing counts of unique values


# In[12]:


train_set['occupation'].value_counts() #Returns object containing counts of unique values


# In[13]:


train_set['native_country'].value_counts() #Returns object containing counts of unique values


# ## Data Visualization

# In[14]:


#The distribution of gender in the train_set dataset.  
fig, ax = plt.subplots()  
x = train_set.sex.unique()  
#Counting 'Males' and 'Females' in the dataset  
y = train_set.sex.value_counts()
#Plotting the bar graph  
ax.bar({0:'Male',1:'Feame'}, y)  
ax.set_xlabel('Sex')  
ax.set_ylabel('Count')  
plt.show() 


# In[15]:


#The distribution of gender in the train_set dataset.  
fig, ax = plt.subplots()  
x = test_set.sex.unique()  
#Counting 'Males' and 'Females' in the dataset  
y = test_set.sex.value_counts()  
#Plotting the bar graph  
ax.bar({0:'Male',1:'Female'}, y)  
#ax.bar(x, y)  
ax.set_xlabel('Sex')  
ax.set_ylabel('Count')  
plt.show() 


# In[16]:


#Bar plot for males and females with wage_class less than 50K tend to work more per week.   
sns.barplot(train_set.sex, train_set['hours_per_week'], hue=train_set['wage_class'])  
plt.show()


# In[17]:


#Bar plot for males and females with wage_class less than 50K tend to work more per week.   
sns.barplot(test_set.sex, test_set['hours_per_week'], hue=test_set['wage_class'])  
plt.show()


# In[18]:


#Creating a box plot for train_set
fig, ax = plt.subplots(figsize=(16, 9))  
sns.boxplot(x='relationship', y='hours_per_week', hue='wage_class', data=train_set, ax=ax)  
ax.set_title('Wage class of people based on relationship and hours_per_week')  
plt.show()


# In[19]:


#Creating a box plot for test_set
fig, ax = plt.subplots(figsize=(16, 9))  
sns.boxplot(x='relationship', y='hours_per_week', hue='wage_class', data=test_set, ax=ax)  
ax.set_title('Wage class of people based on relationship and hours_per_week')  
plt.show()


# In[20]:


#Creating a scatter plots for all pairs of variables of train_set.
pg = sns.PairGrid(data=train_set, hue='wage_class')
pg.map(plt.scatter) 


# In[21]:


#Creating a scatter plots for all pairs of variables of test_set.
pg = sns.PairGrid(data=test_set, hue='wage_class')
pg.map(plt.scatter)


# In[22]:


#Using Strip plot to visualize the train_set.  
fig, ax= plt.subplots(figsize=(16, 9))  
sns.stripplot(train_set['wage_class'], train_set['hours_per_week'], jitter=True, ax=ax)  
ax.set_title('Strip plot')  
plt.show()  


# In[23]:


#Using Strip plot to visualize the test_set.  
fig, ax= plt.subplots(figsize=(16, 9))  
sns.stripplot(test_set['wage_class'], test_set['hours_per_week'], jitter=True, ax=ax)  
ax.set_title('Strip plot')  
plt.show()  


# ## Data Cleaning

# ### Replacing ' ?' with NaN

# In[24]:


train_set_new = train_set.replace({' ?':np.nan}) #Replaces ' ?' with NaN


# In[25]:


train_set_new.isnull().values.any() #Check for any NA’s in the dataframe.


# In[26]:


train_set_new.isnull().sum() #Checks null values in dataframe and prints the sum


# ### Replcaing null values with 'unknown'

# In[27]:


train_set_new.fillna('unknown', inplace=True) #Replaces null values with 'unknown'


# In[28]:


train_set_new.isnull().values.any() #Check for any NA’s in the dataframe.


# In[29]:


train_set_new.isnull().sum()  #Checks null values in dataframe and prints the sum column wise


# In[30]:


test_set_new = test_set.replace({' ?':np.nan}) #Replaces ' ?' with NaN


# In[31]:


test_set_new.isnull().values.any() #Check for any NA’s in the dataframe.


# In[32]:


test_set_new.isnull().sum() #Checks null values in dataframe and prints the sum column wise


# In[33]:


test_set_new.fillna('unknown', inplace=True) #Replaces null values with 'unknown'


# In[34]:


test_set_new.isnull().values.any() #Check for any NA’s in the dataframe.


# In[35]:


test_set_new.isnull().sum() #Checks null values in dataframe and prints the sum column wise


# In[36]:


train_set_new.head() #Returns the first 5 rows of train_set_new dataframe


# In[37]:


test_set_new.head() #Returns the first 5 rows of test_set_new dataframe


# ## Data Pre-Processing 

# ### Creating Features and Targets

# In[38]:


#Removes column that I am trying to predict ('wage_class') from features list
train_features = train_set_new.drop('wage_class', axis=1)

#Creates train labels list
train_labels = (train_set_new['wage_class'] == ' >50K')


# In[39]:


train_features.head() #Returns the first 5 rows of train_features dataframe


# In[40]:


train_labels.unique() #Returns unique values of train_labels.


# In[41]:


#Remove column that I am trying to predict ('wage_class') from features list
test_features = test_set_new.drop('wage_class', axis=1)

#Creates training labels list
test_labels = (test_set_new['wage_class'] == ' >50K.')


# In[42]:


test_features.head() #Returns the first 5 rows of test_features dataframe


# In[43]:


test_labels.unique() #Return unique values of test_labels.


# In[44]:


#Categorical columns contain data that need to be turned into numerical values before being used by XGBoost
CATEGORICAL_COLUMNS = (
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native_country'
)


# In[45]:


CATEGORICAL_COLUMNS


# In[46]:


#Converts data in categorical columns to numerical values
encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}

for col in CATEGORICAL_COLUMNS:
    train_features[col] = encoders[col].fit_transform(train_features[col])
    test_features[col] = encoders[col].fit_transform(test_features[col])    


# In[47]:


train_features.head() #Returns the first 5 rows of train_features dataframe


# In[48]:


test_features.head() #Returns the first 5 rows of test_features dataframe


# In[49]:


#Loads data into DMatrix object
dtrain = xgb.DMatrix(train_features, train_labels)
dtest = xgb.DMatrix(test_features)


# In[50]:


print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))


# In[51]:


print("Train possible labels: ")
print(np.unique(dtrain.get_label()))

print("\nTest possible labels: ")
print(np.unique(dtest.get_label()))


# ## Creating and Training the Model

# ### Specify training parameters

# In[52]:


#Specifies general training parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1,
    'n_estimators': 5
}


# ### Training classifier

# In[53]:


bst = XGBClassifier(**params).fit(train_features, train_labels)


# In[54]:


#bst=xgb.train(params,dtrain)
#xgb.plot_importance(bst)
#ans = bst.predict(dtest)
xgb.plot_importance(bst)
#plt.show()


# In[55]:


#Tree Plot of XGBoost Model for 1st boosted tree
xgb.plot_tree(bst, num_trees=0, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(15, 15)


# In[56]:


#Tree Plot of XGBoost Model for 2nd boosted tree
xgb.plot_tree(bst, num_trees=1, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(15, 15)


# In[57]:


#Tree Plot of XGBoost Model for 3rd boosted tree
xgb.plot_tree(bst, num_trees=2, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(15, 15)


# ### Make predictions

# In[58]:


preds_bst = bst.predict(test_features)
preds_bst


# ### Calculate obtained error

# In[59]:


correct = 0

for i in range(len(preds_bst)):
    if (test_labels[i] == preds_bst[i]):
        correct += 1
        
acc = accuracy_score(test_labels, preds_bst)

print('Predicted correctly: {0}/{1}'.format(correct, len(preds_bst)))
print('Error: {0:.4f}'.format(1-acc))
print('Accuracy: {0:.2f}'.format(correct/len(preds_bst)))


# ## Evaluate results

# In[60]:


#Shows confusion matrix for actual and predicted values for wage_class
confusion_matrix(test_labels,preds_bst)


# In[61]:


#Shows accuracy score of the model
accuracy_score(test_labels,preds_bst)


# ## Problem 1: Prediction task is to determine whether a person makes over 50K a year.
# 
# 
# 

# In[62]:


#Person is 25 years old, working as private job as Machine-op-inspct, Person is male, his education_num is 7
#He has passed 11th, his race is black, his capital gain and capital loss is 0,he Never-married
#He works 40 hours per week and his native country is United-States, his fnlwgt is 226802

wage_class_pred = bst.predict(pd.DataFrame(np.array([[25, 3, 226802, 1, 7, 4, 6, 3, 2, 1, 0, 0, 40, 37]]), columns=test_features.columns))
print("\nThe predicted wage_class is:",wage_class_pred[0] )

if wage_class_pred[0] == True:
  print("\nThis person makes over 50K a year.")
else:
  print("\nThis person doesn't makes over 50K a year.")


# In[63]:


#Person is 35 years old, working as Self-emp-inc job as Exec-managerial, Person is male, his education_num is 13
#He has passed Bachelors, his race is white, his capital gain and capital loss is 0, his marital_status is Married-civ-spouse and he is husband
#He works 60 hours per week and his native country is United-States, his fnlwgt is 182148

wage_class_pred = bst.predict(pd.DataFrame(np.array([[35,	4,	182148,	9,	13,	2,	3,	0,	4,	1,	0,	0,	60,	37]]), columns=test_features.columns))
print("\nThe predicted wage_class is:",wage_class_pred[0] )

if wage_class_pred[0] == True:
  print("\nThis person makes over 50K a year.")
else:
  print("\nThis person doesn't makes over 50K a year.")


# ## Problem 2: Which factors are important

# In[64]:


bst.feature_importances_


# In[65]:


#Barplot
plt.bar(range(len(bst.feature_importances_)), bst.feature_importances_)
plt.show()


# In[66]:


#Feature Importance Flot
xgb.plot_importance(bst)


# ## Problem 3: Which algorithms are best for this dataset

# In[67]:


#Creating an empty dictionary
algorithm_score_dict ={}


# ### Logistic Regression
# 
# 

# In[68]:


#Instantiating a logistic regression model, and fit with train_features and train_labels
lgr = LogisticRegression()
lgr.fit(train_features,train_labels)

#Predicting the wage_class using test_features
pred_lgr = lgr.predict(test_features)


# In[69]:


#Create and show confusion matrix for actual and predicted values of wage_class
confusion_matrix(test_labels,pred_lgr)


# In[70]:


#Calculate and shows accuracy score of the model 
accuracy_score_lgr = accuracy_score(test_labels,pred_lgr)
accuracy_score_lgr


# In[71]:


#Adding accuracy score of Logistic Regression into algorithm_score_dict
algorithm_score_dict.update({"Logistic Regression":accuracy_score_lgr})


# ### Decision Tree

# In[72]:


#Instantiating a decision tree classifier, and fit with train_features and train_labels
dtc = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
dtc.fit(train_features, train_labels)

#Predicting the wage_class using test_features
pred_dtc = dtc.predict(test_features)


# In[73]:


#Create and show confusion matrix for actual and predicted values of wage_class
confusion_matrix(test_labels,pred_dtc)


# In[74]:


#Calculate and shows accuracy score of the model 
accuracy_score_dtc = accuracy_score(test_labels,pred_dtc)
accuracy_score_dtc


# In[75]:


#Adding accuracy score of Decision Tree into algorithm_score_dict
algorithm_score_dict.update({"Decision Tree":accuracy_score_dtc})


# ### Random Forest

# In[76]:


#Instantiating a random forest classifier, and fit with train_features and train_labels
rfc = RandomForestClassifier()
rfc.fit(train_features,train_labels)

#Predicting the wage_class using test_features
pred_rfc = rfc.predict(test_features)


# In[77]:


#Create and show confusion matrix for actual and predicted values of wage_class
confusion_matrix(test_labels,pred_rfc)


# In[78]:


#Calculate and shows accuracy score of the model 
accuracy_score_rfc = accuracy_score(test_labels,pred_rfc)
accuracy_score_rfc


# In[79]:


#Adding accuracy score of Random Forest into algorithm_score_dict
algorithm_score_dict.update({"Random Forest":accuracy_score_rfc})


# ### XGBoost

# In[80]:


#Instantiating a XGBoost classifier, and fit with train_features and train_labels
xgbst = XGBClassifier(**params).fit(train_features, train_labels)

#Predicting the wage_class using test_features
pred_xgbst = xgbst.predict(test_features)


# In[81]:


#Create and show confusion matrix for actual and predicted values of wage_class
confusion_matrix(test_labels,pred_xgbst)


# In[82]:


#Calculate and shows accuracy score of the model 
accuracy_score_xgbst = accuracy_score(test_labels,pred_xgbst)
accuracy_score_xgbst


# In[83]:


#Adding accuracy score of Random Forest into algorithm_score_dict
algorithm_score_dict.update({"XGBoost":accuracy_score_xgbst})


# In[84]:


#Shows the algorithm_score_dict data
algorithm_score_dict


# In[85]:


print("The best algorithm for this dataset is :", max(algorithm_score_dict.items(), key=operator.itemgetter(1))[0])
  

