import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
    print(a,b,c)
    X = X.reshape(a, b*c)
    return X, y

# load mnist only for classes 0 and 8
train_images, train_labels = load_mnist([0,8], 200)
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.3, random_state=256)

# generate a random index for the test image
random_index = np.random.randint(0, X_test.shape[0])

import matplotlib.pyplot as plt
x_test = X_test[random_index].reshape(28,28)
plt.imshow(x_test)
plt.show()


from PIL import Image


# Convert the array to a PIL image and save as grayscale JPEG
img = Image.fromarray(x_test, mode="L")
img.save("output.jpg")