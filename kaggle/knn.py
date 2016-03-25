import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import cross_val_score


#read data 
dataset = pd.read_csv("./train.csv")
X_train = dataset.values[0:, 1:]
y_train = dataset.values[0:, 0]


#for fast evaluation
X_train_small = X_train[:1000, :]
y_train_small = y_train[:1000]


X_test = pd.read_csv("./test.csv").values
#knn
from sklearn.neighbors import KNeighborsClassifier
#begin time
start = time.clock()

#progressing
knn_clf=KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, X_train_small, y_train_small, cv=3)

print( score.mean() )
#end time
elapsed = (time.clock() - start)
print("Time used:",int(elapsed), "s")

clf=knn_clf

start = time.clock()
clf.fit(X_train_small,y_train_small)
elapsed = (time.clock() - start)
print("Training Time used:",int(elapsed/60) , "min")

result=clf.predict(X_test)
result = np.c_[range(1,len(result)+1), result.astype(int)]
df_result = pd.DataFrame(result, columns=['ImageId', 'Label'])

df_result.to_csv('./results.knn.csv', index=False)
#end time
elapsed = (time.clock() - start)
print("Test Time used:",int(elapsed/60) , "min")