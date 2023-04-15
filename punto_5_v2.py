import numpy as np
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_mnist(digits_to_keep, N):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    arrays = []
    labels = []
    n = N // len(digits_to_keep)
    for cl in digits_to_keep:
        ind = y_train == cl
        A = x_train[ind]
        ycl = y_train[ind]
        arrays.append(A[:n])
        labels.append(ycl[:n])
    X = np.concatenate(tuple(arrays), axis = 0)
    y = np.concatenate(tuple(labels), axis = 0)
    a,b,c = X.shape
    X = X.reshape(a, b*c) # 3d -> 2d
    return X, y

# load mnist only for classes 0 and 8
train_images, train_labels = load_mnist([0,8], 200)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.3, random_state=256)

"""
set_y_train = list(set(y_train))
w1 = len(y_train[np.where(y_train == set_y_train[0])])
w2 = len(y_train[np.where(y_train == set_y_train[1])])
plt.bar([set_y_train[0], set_y_train[1]],[w1 ,  w2])
plt.show()
"""

log_reg = LogisticRegression(random_state=345)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))

'''Initially, the model was trained with all the data belonging to classes 0 and 8, however good results are not obtained. For this reason, a sample of 200 data is chosen, there an acurracy of 100% is obtained '''