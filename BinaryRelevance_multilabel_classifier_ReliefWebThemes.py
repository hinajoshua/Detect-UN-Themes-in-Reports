# MULTILABEL TEXT CLASSIFIER MODEL BUILDING FOR THEME DETECTION
# @author: Hina Joshua

import pandas as pd
import numpy as np
import pickle
import joblib

### Split Dataset into Train and Test
from sklearn.model_selection import train_test_split
# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
# ML Pkgs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
# Multi Label Pkgs
from skmultilearn.problem_transform import BinaryRelevance


#load dataset
df = pd.read_csv(r'rw_themes_balanced_df.csv')

themes = ['Gender',
       'Peacekeeping and Peacebuilding', 'Agriculture', 'Food and Nutrition',
       'Contributions', 'Coordination', 'Health', 'Water Sanitation Hygiene',
       'Protection and Human Rights', 'Mine Action',
       'Shelter and Non-Food Items', 'Climate Change and Environment',
       'Logistics and Telecommunications', 'Recovery and Reconstruction',
       'Safety and Security', 'Education', 'Humanitarian Financing',
       'Disaster Management', 'HIV/Aids',
       'Camp Coordination and Camp Management']

# #reduce dataset size for testing
# df = df.sample(10000)

#Tfidf vectorizer
vectorizer = TfidfVectorizer(stop_words='english', 
                             min_df=0.01, 
                             max_df=0.8, 
                             max_features=10000, 
                             ngram_range=(1, 2))

Xfeatures = vectorizer.fit_transform(df.text).toarray()
y = df[themes].astype(int)

print(f"The number of features = {Xfeatures.shape[1]}")
print(f"The number of documents = {Xfeatures.shape[0]}")

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(Xfeatures,y,test_size=0.2,random_state=42)

#### Binary Relevance classficiation
#Convert Our Multi-Label Prob to Multi-Class

# #Logistic Regression Estimator
# max_iter = 1000
# solver = 'newton-cg' #'lbfgs'
# C=9
# binary_rel_clf = BinaryRelevance(LogisticRegression(solver=solver,
#                                                     C=C,
#                                                     max_iter=max_iter,
#                                                     multi_class='auto'))
# binary_rel_clf.fit(X_train,y_train)
# # Predictions Logistic Regression
# br_prediction = binary_rel_clf.predict(X_test)

# #F1 Score LR
# lr_f1 = f1_score(y_test, br_prediction, average='micro')
# print(f"Logistic Regression F1 Score = {lr_f1}")

#Linear SCV
binary_rel_svc = BinaryRelevance(LinearSVC(C=9))
binary_rel_svc.fit(X_train,y_train)

# Predictions linearsvc
svc_prediction = binary_rel_svc.predict(X_test)

#F1 Score SVC
svc_f1 = f1_score(y_test, svc_prediction, average='micro') 
print(f"Linear SVC F1 Score = {svc_f1}")

# #Results
# The number of features = 4890
# The number of documents = 24329
# Logistic Regression F1 Score = 0.8689281266026815
# Linear SVC F1 Score = 0.8819053452378147

#Save Model
filename = 'Model_RW_ThemeDetect.pkl'
# pickle.dump(binary_rel_svc, open(filename, 'wb'))
joblib.dump(binary_rel_svc, filename)

#Save Vectorizer
vec_name = 'Vectorizer_RW_ThemeDetect.pkl'
joblib.dump(vectorizer, vec_name)
# with open(vec_name, 'wb') as fin:
#   pickle.dump(vectorizer, fin)

