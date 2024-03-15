# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:46:06 2024

@author: Ashish Chincholikar
Naive Bayes Assignments
"""


"""
1. Business Problem

1.1 what is business objective?
1.2 Are there any constraints

2. Create a Data Dictionary 

name of feature 
description 
type 
relevance

3. Data Preprocessing

3.1 Data Cleaning , feature engineering ,etc

4. Exploratory Data Analysis(EDA)

4.1 Summary.
4.2 Univariate analysis.
4.3 Bivariate analysis.


5. Model Building 

5.1 Build the model on the scaled data(try multiple models)
5.2 Build a Naive bayes model.
5.3 Validate the model with test data and obtain a confusion matrix , get precision ,recall and acurate it
5.4 tune the model and imporve the accuracry


6.Write down the benifits/impact of the solution - in what way does the business(client)benfit from the solution 
provided?

"""


"""
1. Business Problem
    ~In today's Day and time , organizations have very large workforce and provide employeement to them
    with it they also need to maintaint their individual record details such as name , number ,job-role,
    salary , education.
    ~in such a scenrio there is need to develope a model to Classify the Salary of the employees and draw
    some insightful information from this classsification
    
1.1 what is business objective?
    ~To understand the Employee data w.r.t various feature with the Salary
    ~TO develope a classification model to classify the Salaries of the employees
    
1.2 Are there any constraints
    ~
    ~

2. Create a Data Dictionary 
name of feature 
description 
type 
relevance

1. age , age of a person , nominal data , Relavent data
2. workclass , A work class is a grouping of work , ~ , Relavent data
3. education , Education of an individual , ~  , Relevant data
4. Education number , Education number of an individual ,~ , irrelevant
5. marital status , marital status of an individula , ~ , irreleavant
6. Occupation , Occupation of an individual , ~ , relevant
7. Relationship , ~ , ~ , irrelevant
8. Race , Race of an individual , ~ , irrelevant
9. sex  , gender of an individual , ~ , irrelevant
10. capitalGain , profit recived from the scale of an investment , ~ , relevant
11. capitalloss , A decrease in the value of a capital asset ,~ , relevant
12. number of hours work per week , ~ , relevant
13. Native , country of birth of individual , ~ , less relevant
14. Salary , Salary of the Individual , 
"""

""" 
3. Data Preprocessing
3.1 Data Cleaning , feature engineering ,etc
"""

import pandas as pd

email_data = pd.read_csv("C:/Supervised_ML/Naive_Bayes_Algo/SalaryData_Train.csv")

#step1 -  perform the operations related to EDA 
#how many data-points and features
email_data.shape
#(30161, 14)

#What are the Column Names in our dataset
email_data.columns

#datatypes
email_data.dtypes

""" 
age               int64
workclass        object
education        object
educationno       int64
maritalstatus    object
occupation       object
relationship     object
race             object
sex              object
capitalgain       int64
capitalloss       int64
hoursperweek      int64
native           object
Salary           object
dtype: object
"""

#---------------------------------------------------------------
#identifying the duplicates-->and then drop it

#outlier treatment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer



sns.boxplot(email_data.capitalgain)
#there are outliers in capital gain column 


# education no , marital status ,race , native can be dropped

# Correlation analysis
correlation_matrix = email_data.corr()
print(correlation_matrix)

corr=email_data.corr()
sns.heatmap(corr)
plt.title('Correlation Heatmap')
plt.show()


##########cleaning of data 
import re 

def cleaning_text(i):
    w=[]
    i = re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
#### Testing above function with some test text \

email_data.maritalstatus = email_data.maritalstatus.apply(cleaning_text)
email_data = email_data.loc[email_data.maritalstatus != "",:]
from sklearn.model_selection import train_test_split 
email_train = email_data
email_test = pd.read_csv("C:/Supervised_ML/Naive_Bayes_Algo/SalaryData_Test.csv")
#creating matrix of token counts for entire text document

def split_into_words(i):
    return [word for word in i.split(" ")]

emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.maritalstatus)
all_emails_matrix = emails_bow.transform(email_data.maritalstatus)
#for training messages 
train_emails_matrix = emails_bow.transform(email_train.maritalstatus)
#for testing messages 
test_emails_matrix = emails_bow.transform(email_test.maritalstatus)
#Learing Term weightaging and normaling on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)
#Preparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
#preparing TFIDF for test mails 
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape
 
####Now let us apply this to Naive Bayes 

from sklearn.naive_bayes import MultinomialNB as MB 
classifier_mb = MB()
classifier_mb.fit(train_tfidf,email_train.Salary)

#evaluation on test data 
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test.Salary)
accuracy_test_m


#--------------------------------------------------------------------------------------
#1.1. Business Objective:
'''
The primary objective is to prepare a classification model using 
the Naive Bayes algorithm for the salary dataset.
'''

#1.2. Constraints:
'''
There are no specific constraints mentioned in the problem statement,
 so we can assume that the main goal is to build an accurate
 classification model without any specific limitations.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
email_data=pd.read_csv("C:/Supervised_ML/Naive_Bayes_Algo/Disaster_tweets_NB.csv",encoding="ISO-8859-1")


#step1 -  perform the operations related to EDA 
#how many data-points and features
email_data.shape
# (7613, 5)

#What are the Column Names in our dataset
email_data.columns
#Index(['id', 'keyword', 'location', 'text', 'target'], dtype='object')

#datatypes
email_data.dtypes
"""
id           int64
keyword     object
location    object
text        object
target       int64
dtype: object
"""
#droping Null values
email_data.isnull().sum()
#There are some null values in the data for keyword and location
email_data['location'].fillna(value='unknown', inplace=True)
email_data['keyword'].fillna(value='missing', inplace=True)
email_data.isnull().sum()#

#EDA
email_data.info()
email_data.describe()
#This is a very unstructuted Data set we will have to clean the data first
import re
def cleaning_text(i):
    w=[]
    i=re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

email_data.keyword=email_data.keyword.apply(cleaning_text)
email_data=email_data.loc[email_data.keyword!="",:]

email_data.location=email_data.location.apply(cleaning_text)
email_data=email_data.loc[email_data.keyword!="",:]

email_data.text=email_data.text.apply(cleaning_text)
email_data=email_data.loc[email_data.text!="",:]

email_data.isnull().sum()

email_data['keyword'].value_counts().head(10)

#Boxplot
sns.boxplot(data=email_data,x=email_data['keyword'].value_counts())
plt.title('Boxplot of keyword')
plt.xlabel('Keyword')
plt.ylabel('Values')
plt.show()

# Correlation analysis
correlation_matrix = email_data.corr()
print(correlation_matrix)
corr=email_data.corr()
sns.heatmap(corr)
plt.title('Correlation Heatmap')
plt.show()


from sklearn.model_selection import train_test_split
email_train,email_test=train_test_split(email_data,test_size=0.2)

def split_into_words(i):
    return [word for word in i.split(" ")]


emails_bow=CountVectorizer(analyzer=split_into_words).fit(email_data.keyword)
all_emails_matrix=emails_bow.transform(email_data.keyword)

train_emails_matrix=emails_bow.transform(email_train.keyword)
test_emails_matrix=emails_bow.transform(email_test.keyword)

tfidf_Transformer=TfidfTransformer().fit(all_emails_matrix)
train_tfidf=tfidf_Transformer.transform(train_emails_matrix)
test_tfidf=tfidf_Transformer.transform(test_emails_matrix)

test_tfidf.shape


from sklearn.naive_bayes import MultinomialNB as MB
classifer_mb=MB()
classifer_mb.fit(train_tfidf,email_train.location)

test_pred_m=classifer_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.location)
accuracy_test_m
###############################################################
# Data Dictionary
"""1.User ID:ID of an User | Not Relevant
2.Gender: Gender of an individual | Relevant
3.Age: Age of a person | Relevant
4.EstimatedSakary: Salary | Relevant
5.Purchased: Purchased Car or Not | Relevant(Label)
"""
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
df = pd.read_csv("C:/Supervised_ML/Naive_Bayes_Algo/NB_Car_Ad.csv")
df
df.columns


#Here the User ID is irrelevant so we will discard it 
df.drop({'User ID'},inplace=True,axis=1)

from sklearn.model_selection import train_test_split 
df_train,df_test = train_test_split(df,test_size=0.2)

df_bow = CountVectorizer().fit(df.Gender)

df_matrix = df_bow.transform(df.Gender)
#for training messages 
train_df_matrix = df_bow.transform(df_train.Gender)
#for testing messages 
test_df_matrix = df_bow.transform(df_train.Gender)
#Learing Term weightaging and normaling on entire emails
tfidf_transformer = TfidfTransformer().fit(df_matrix)
#Preparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_df_matrix)
#preparing TFIDF for test mails 
test_tfidf = tfidf_transformer.transform(test_df_matrix)
test_tfidf.shape
####Now let us apply this to Naive Bayes 

from sklearn.naive_bayes import MultinomialNB as MB 
classifier_mb = MB()
classifier_mb.fit(train_tfidf,df_train.Gender)

#evaluation on test data 
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == df_test.Gender)
accuracy_test_m













