#%%In[1]
# 1장 시작
#%%In[2]
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
#%%In[3]
from scipy import sparse

# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))
#%%In[4]
# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)  # csc_matrix 이게 희소 행렬을 뜻함
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
#%%In[5]
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))

#%%In[6]
%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")

#%%In[7]
import pandas as pd
# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [24, 13, 53, 33]
        }
data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
display(data_pandas)
#%%In[8]
# Select all rows that have an age column greater than 30
display(data_pandas[data_pandas.Age > 30])

#%%In[9]
import sys
print("Python version: {}".format(sys.version))
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))

#%%In[10]
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#%%In[11]
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

#%%In[12]
print(iris_dataset['DESCR'][:193] + "\n...")

#%%In[13]
print("Target names: {}".format(iris_dataset['target_names']))

#%%In[14]
print("Feature names: \n{}".format(iris_dataset['feature_names']))

#%%In[15]
print("Type of data: {}".format(type(iris_dataset['data'])))

#%%In[16]
print("Shape of data: {}".format(iris_dataset['data'].shape))

#%%In[17]
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

#%%In[18]
print("Type of target: {}".format(type(iris_dataset['target'])))

#%%In[19]
print("Shape of target: {}".format(iris_dataset['target'].shape))

#%%In[20]
print("Target:\n{}".format(iris_dataset['target']))

#%%In[21]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

#%%In[22]
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

#%%In[23]
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#%%In[24]
# 이거 부터 실행해야됨 이거 ㄹㅇ 하나도 모르겠다.
import mglearn
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


#%%In[25]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


#%%In[26]
knn.fit(X_train, y_train)

#%%In[27]
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

#%%In[28]
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))

#%%In[29]
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

#%%In[30]
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#%%In[31]
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

#%%In[32]
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
