import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
dataset = pd.read_csv("./train.csv")

X_train = dataset.values[0:,1:]
y_train = dataset.values[0:,0]

X_test = pd.read_csv("./test.csv").values

X_train_small = X_train[0:10000,:]
y_train_small = y_train[0:10000]

print  "read already"

start = time.clock()

LR_CLF =LogisticRegression(penalty='l2', solver ='lbfgs', multi_class='multinomial', max_iter=800,  C=0.2 )

score = cross_val_score(LR_CLF, X_train_small, y_train_small, cv=3)

print score.mean()

LR_CLF.fit(X_train_small,y_train_small)

print "training time used: %d"%((int)(time.clock()-start)/60)

result = LR_CLF.predict(X_test)

result = np.c_[range(1,len(result)+1), result.astype(int)]

df_result = pd.DataFrame(result, columns=['ImageId', 'Label'])

df_result.to_csv('./results.lr.csv', index=False)

elapsed = (time.clock() - start)

print("Test Time used:",int(elapsed/60) , "min")