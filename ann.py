import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('final_data_frame3.csv')
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

#no feature selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#L1 based feature selection
"""
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.7, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

#Tree based feature selection
"""
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""


#Variance threshold method
"""
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = sel.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

#RFE based feature selection
"""
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
selector.support_
X_new = selector.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 8, epochs = 120)



y_pred = classifier.predict(X_test)
y_pred[0] = ((y_pred[0] > 0.5))
y_pred[1] = ((y_pred[1] > 0.5))
y_pred[2] = ((y_pred[2] > 0.5))
y_pred[3] = ((y_pred[3] > 0.5))
y_final = []
for i in range(len(y_pred)):
    if(y_pred[i][0]>y_pred[i][1]):
        if(y_pred[i][0]>y_pred[i][2]):
            if(y_pred[i][0]>y_pred[i][3]):
                y_final.append(0)
            else:
                y_final.append(3)
        else:
            if(y_pred[i][2]>y_pred[i][3]):
                y_final.append(2)
            else:
                y_final.append(3)
    else:
        if(y_pred[i][1]>y_pred[i][2]):
            if(y_pred[i][1]>y_pred[i][3]):
                y_final.append(1)
            else:
                y_final.append(3)
        else:
            if(y_pred[i][2]>y_pred[i][3]):
                y_final.append(2)
            else:
                y_final.append(3)
y_final=np.array(y_final)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_final)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_final)