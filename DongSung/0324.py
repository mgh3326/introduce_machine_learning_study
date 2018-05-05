# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:12:53 2018

@author: 김동성
"""

import mglearn


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    
    
def bar(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    rate = survived / (survived + dead)
    rate.plot(kind = 'bar', figsize=(10,5))
    
#sex_mapping = {"male": 0, "female": 1}
#train['Sex'] = train['Sex'].map(sex_mapping)


def girlbar(feature):
    survived = train[train['Sex'] == 1][train['Survived']==1][feature].value_counts()
    dead = train[train['Sex'] == 1][train['Survived']==0][feature].value_counts()
    rate = survived / (survived + dead)
    rate.plot(kind = 'bar', figsize=(10,5))
    
girlbar('Pclass')

def boybar(feature):
    survived = train[train['Sex'] == 0][train['Survived']==1][feature].value_counts()
    dead = train[train['Sex'] == 0][train['Survived']==0][feature].value_counts()
    rate = survived / (survived + dead)
    rate.plot(kind = 'bar', figsize=(10,5))

boybar('Pclass')

train[train['Sex'] == 0][train['Pclass'] == 2][train['Survived'] == 1]
train[train['Sex'] == 0][train['Pclass'] == 3][train['Survived'] == 1]

train[train['Age']<17]

a = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare','Embarked','FamilySize','Title']


'남자중 1등석이 아닌자리의 생존자는 대부분 아이였다.'
'Sex,Pclass,Age'

train[train['Sex'] == 0][train['Pclass'] == 2][train['Age']>17][train['Survived'] == 1]



train[train['Sex'] == 0][train['Age']>15][train['Pclass'] == 3][train['Survived']==1]
train[train['Sex'] == 0][train['Age']>15][train['Pclass'] == 1][train['Fare'] < 30]['Survived'].value_counts()


#train[train['Sex'] == 0][train['Age']<=15][train['Pclass'] == 1]['Survived'].value_counts()
train[train['Sex'] == 0][train['Age']>15][train['Pclass'] == 1]['Survived'].value_counts()
#train[train['Sex'] == 0][train['Age']<=15][train['Pclass'] == 2]['Survived'].value_counts()
train[train['Sex'] == 0][train['Age']>15][train['Pclass'] == 2]['Survived'].value_counts()
#train[train['Sex'] == 0][train['Age']<=15][train['Pclass'] == 3]['Survived'].value_counts()
train[train['Sex'] == 0][train['Age']>15][train['Pclass'] == 3]['Survived'].value_counts()


#train[train['Sex'] == 1][train['Age']<=15][train['Pclass'] == 1]['Survived'].value_counts()
train[train['Sex'] == 1][train['Age']>15][train['Pclass'] == 1]['Survived'].value_counts()
#train[train['Sex'] == 1][train['Age']<=15][train['Pclass'] == 2]['Survived'].value_counts()
train[train['Sex'] == 1][train['Age']>15][train['Pclass'] == 2]['Survived'].value_counts()
#train[train['Sex'] == 1][train['Age']<=15][train['Pclass'] == 3]['Survived'].value_counts()
train[train['Sex'] == 1][train['Age']>15][train['Pclass'] == 3]['Survived'].value_counts()



#########################################

train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 4, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
#train_test_data
#train.info()
#################이름끝


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

#train_test_data

#####################성별 맵핑끝
    

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

train.info()


######################나이 중간값 할당.끝
train.loc[train['Embarked'].isnull() == True, 'Embarked'] = 'C'
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

#train_test_data
train.info()

###################### 탑승지,요금 바꾸기


#features_drop = ['Ticket', 'SibSp', 'Parch', 'Cabin', 'Fare', 'Embarked']
features_drop = ['Ticket','Cabin']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)


train.info()
test.info()
train.head()

train_test_data = [train,test]
#train_test_data
####################### 특징 없애기 끝






train.info()
dataframe = train.drop('PassengerId', axis = 1)
dataframe.info()


features_drop2 = ['SibSp','Parch']
dataframe = dataframe.drop(features_drop2, axis = 1)
dataframe.info()


xdf = dataframe.drop('Survived', axis = 1)
xdf = xdf.drop('Embarked', axis = 1)
ydf = dataframe['Survived']


pd.plotting.scatter_matrix(xdf, c=ydf, figsize=(20,20), marker='o', hist_kwds={'bins':20}, s=30, alpha=.8, cmap=mglearn.cm3)





















#############################상관관계를 알아보자
'아가 1,2등석은 다산거로하자. 4개분류'
'어른남자 2,3등석은 다죽은거다. 2개분류'
'어른여자 1,2등석은 다산거다. 2개분류'

'아가 3등석은 반은살고 반은 죽은거로 찍자. 2개분류'
'어른남자 1등석은 삼분의 일 살았다. knn이든 뭐든해보자 1개분류'
'어른여자 3등석은 반살고반죽음. 1개분류'


train.loc[(train['Sex'] == 0) & (train['Age']<=15) & (train['Pclass'] == 1), 'pre1'] = 1
train.loc[(train['Sex'] == 0) & (train['Age']<=15) & (train['Pclass'] == 2), 'pre1'] = 1
train.loc[(train['Sex'] == 1) & (train['Age']<=15) & (train['Pclass'] == 1), 'pre1'] = 1
train.loc[(train['Sex'] == 1) & (train['Age']<=15) & (train['Pclass'] == 2), 'pre1'] = 1

train.loc[(train['Sex'] == 0) & (train['Age']>15) & (train['Pclass'] == 2), 'pre1'] = 0
train.loc[(train['Sex'] == 0) & (train['Age']>15) & (train['Pclass'] == 3), 'pre1'] = 0

train.loc[(train['Sex'] == 1) & (train['Age']>15) & (train['Pclass'] == 1), 'pre1'] = 1
train.loc[(train['Sex'] == 1) & (train['Age']>15) & (train['Pclass'] == 2), 'pre1'] = 1

train.info()

#tr = train[train['pre1'].isnull() == True]
#tr.info()

#tr = train[train['pre1'].isnull() == False]
#len(tr[tr['Survived'] != tr['pre1']])
#len(tr)

######################################
#train.info()
#test.info()
#train.shape
#test.shape
#train_test_data



#나이 죽이기
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

#요금 죽이기
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

train.head()






#######################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


tr = train[train['pre1'].isnull() == True]
tr.info()
x = tr.drop('Survived', axis=1)
x.info()
x = x.drop('pre1', axis=1)
x.info()
y = tr['Survived']

#knn
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score1 = cross_val_score(clf, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score1)
round(np.mean(score1)*100,2)

for i in range(5,20):
    clf = KNeighborsClassifier(n_neighbors = i)
    score1 = cross_val_score(clf, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
    print(np.mean(score1)*100,2)


#Decision Tree
tree = DecisionTreeClassifier()
scoring = 'accuracy'
score2 = cross_val_score(tree, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score2)
round(np.mean(score2)*100,2)

for i in range(2,13):
    tree = DecisionTreeClassifier(max_depth=i)
    score2 = cross_val_score(tree, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
    print(np.mean(score2)*100,2)
#Random Forest
forest = RandomForestClassifier(n_estimators=10)
scoring = 'accuracy'
score3= cross_val_score(forest, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score3)
round(np.mean(score3)*100,2)

for i in range(2,20):
    forest = RandomForestClassifier(n_estimators=i)
    score3= cross_val_score(forest, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
    print(np.mean(score3)*100,2)
#Naive Bayes
NB = GaussianNB()
scoring = 'accuracy'
score4 = cross_val_score(NB, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score4)
round(np.mean(score4)*100,2)


#SVM
svm = SVC()
scoring = 'accuracy'
score5 = cross_val_score(svm, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score5)
round(np.mean(score5)*100,2)



########################################################머신러닝 교재


from sklearn.model_selection import train_test_split
#knn
X_train, X_test, y_train, y_test = train_test_split(x,y, random_state = 42)



for i in range(5,15):
    clf = KNeighborsClassifier(n_neighbors = i)
    clf.fit(X_train, y_train)
    print('Neighbor = ',i)
    print(clf.score(X_train, y_train))    
    print(clf.score(X_test, y_test))
    print('-----------------------------')
    


#결정트리

tree = DecisionTreeClassifier(max_depth=6)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
tree.score(X_test, y_test)

#for i in range(10):
#    print(i)

x.info()


a = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare','Embarked','Title']
for i in range(9):
    print(a[i])
    print(tree.feature_importances_[i])
    



#랜덤포레스트


forest = RandomForestClassifier(n_estimators=12)
forest.fit(X_train, y_train)
forest.score(X_train, y_train)
forest.score(X_test, y_test)

for i in range(2,20):
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(X_train, y_train)
    print(i)
    print(forest.score(X_train, y_train))
    print(forest.score(X_test, y_test))

for i in range(9) :
    print(a[i])
    print(forest.feature_importances_[i])



#SVM
    
    
svm = SVC(kernel='rbf', C=10, gamma=10).fit(X_train, y_train)
svm.score(X_train, y_train)
svm.score(X_test, y_test)

for j in [0.1, 1, 1000]:
    for k in [0.1, 1, 10]:
        svm = SVC(kernel='rbf', C=j, gamma=k).fit(X_train, y_train)
        print('c = ',j,'gamma = ',k)
        print(svm.score(X_train, y_train))
        print(svm.score(X_test, y_test))




#############################################
#Testing
    
clf = SVC()
clf.fit(x, y)

NB.fit(x,y)

forest.fit(x,y)
#x.info()
#NBB.predict(x)
test.info()
x.info()
test_data = x.copy()
test_data.info()

prediction = clf.predict(test_data)

prediction2 = NB.predict(test_data)

prediction3 = forest.predict(test_data)

x['Survived'] = prediction3
x.info()

train.loc[train['PassengerId'] == x['PassengerId'], 'pre1'] = x['Survived']

train.info()

train.loc[train['pre1'].isnull() == True, 'pre1'] = prediction3

train[train['Survived'] != train['pre1']][['Survived','Pclass','Sex','Age','Title']]
len(train[train['Survived'] != train['pre1']])
#x['Survived']    
train.info()    
5600/891

    
#tr2 = train[train['pre1'].isnull() == False]
#tr.info()
x = tr.drop('Survived', axis=1)
x.info()
x = x.drop('pre1', axis=1)
x.info()
y = tr['Survived']
    
############################################
test.info()    

test.loc[(test['Sex'] == 0) & (test['Age']<=15) & (test['Pclass'] == 1), 'Survived'] = 1
test.loc[(test['Sex'] == 0) & (test['Age']<=15) & (test['Pclass'] == 2), 'Survived'] = 1
test.loc[(test['Sex'] == 1) & (test['Age']<=15) & (test['Pclass'] == 1), 'Survived'] = 1
test.loc[(test['Sex'] == 1) & (test['Age']<=15) & (test['Pclass'] == 2), 'Survived'] = 1

test.loc[(test['Sex'] == 0) & (test['Age']>15) & (test['Pclass'] == 2), 'Survived'] = 0
test.loc[(test['Sex'] == 0) & (test['Age']>15) & (test['Pclass'] == 3), 'Survived'] = 0

test.loc[(test['Sex'] == 1) & (test['Age']>15) & (test['Pclass'] == 1), 'Survived'] = 1
test.loc[(test['Sex'] == 1) & (test['Age']>15) & (test['Pclass'] == 2), 'Survived'] = 1



#tr = train[train['pre1'].isnull() == True]
#tr.info()
tt = test[test['Survived'].isnull() == True]
tt.info()
X = tt.drop('Survived',axis=1)
prediction = forest.predict(X)
test.loc[test['Survived'].isnull() == True, 'Survived'] = prediction
test.info()
test.head(50)
#####################################################
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": test['Survived']})
submission.to_csv('submission2.csv', index=False)
submission = pd.read_csv('submission2.csv')
submission.head(20)
#####################################제출하자!
#Y = tt['Survived']
#x = tr.drop('Survived', axis=1)
#x.info()
#x = x.drop('pre1', axis=1)
#x.info()
#y = tr['Survived']
    
    
test.to_csv('test.csv', index=False)


test[test['Sex'] == 0][test['Age'] <= 15][test['Pclass'] == 1]['Survived']
test[test['Sex'] == 0][test['Age'] <= 15][test['Pclass'] == 2]['Survived']
test[test['Sex'] == 1][test['Age'] <= 15][test['Pclass'] == 1]['Survived']
test[test['Sex'] == 1][test['Age'] <= 15][test['Pclass'] == 2]['Survived']    
    
test[test['Sex'] == 0][test['Age'] > 15][test['Pclass'] == 2]['Survived']    
test[test['Sex'] == 0][test['Age'] > 15][test['Pclass'] == 3]['Survived']        


