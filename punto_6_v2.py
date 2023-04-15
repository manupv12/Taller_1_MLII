import numpy as np
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from unsupervised import SVD
from unsupervised import PCA
from unsupervised import TSNE

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
    X = X.reshape(a, b*c)
    return X, y

# load mnist only for classes 0 and 8
train_images, train_labels = load_mnist([0,8], 100)



X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.3, random_state=256)

# ===================================

unsupervised_svd = SVD(n_components=2)
unsupervised_svd.fit(X_train)
X_train_unsupervised_svd = unsupervised_svd.transform(X_train)
X_test_unsupervised_svd = unsupervised_svd.transform(X_test)

plt.scatter(X_train_unsupervised_svd[:, 0], X_train_unsupervised_svd[:, 1], c=y_train, cmap='viridis', s=5)
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.colorbar()
plt.show()

"""
X_train_unsupervised_pca, X_test_unsupervised_pca, y_train_unsupervised_pca, y_test_unsupervised_pca =train_test_split(X_train_unsupervised_pca, 
                                                                                                     y_train,
                                                                                                 test_size=0.3, 
                                                                                                 random_state=128)
"""

log_reg_with_unsupervised_svd = LogisticRegression(random_state=42)
log_reg_with_unsupervised_svd.fit(X_train_unsupervised_svd, y_train)
y_pred_unsupervised_svd = log_reg_with_unsupervised_svd.predict(X_test_unsupervised_svd)
print(classification_report(y_test, y_pred_unsupervised_svd))


# ===================================

unsupervised_pca = PCA(n_components=2)
unsupervised_pca.fit(X_train)
X_train_unsupervised_pca = unsupervised_pca.transform(X_train)
X_test_unsupervised_pca = unsupervised_pca.transform(X_test)

plt.scatter(X_train_unsupervised_pca[:, 0], X_train_unsupervised_pca[:, 1], c=y_train, cmap='viridis', s=5)
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.colorbar()
plt.show()

"""
X_train_unsupervised_pca, X_test_unsupervised_pca, y_train_unsupervised_pca, y_test_unsupervised_pca =train_test_split(X_train_unsupervised_pca, 
                                                                                                     y_train,
                                                                                                 test_size=0.3, 
                                                                                                 random_state=128)
"""

log_reg_with_unsupervised_pca = LogisticRegression(random_state=42)
log_reg_with_unsupervised_pca.fit(X_train_unsupervised_pca, y_train)
y_pred_unsupervised_pca = log_reg_with_unsupervised_pca.predict(X_test_unsupervised_pca)
print(classification_report(y_test, y_pred_unsupervised_pca ))

# ===================================

unsupervised_TSNE = TSNE(n_components=2)
#unsupervised_TSNE.fit_transform(X_train)
X_train_unsupervised_TSNE = unsupervised_TSNE.fit_transform(X_train)
X_test_unsupervised_TSNE = unsupervised_TSNE.fit_transform(X_test)

plt.scatter(X_train_unsupervised_TSNE[:, 0], X_train_unsupervised_TSNE[:, 1], c=y_train, cmap='viridis', s=5)
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.colorbar()
plt.show()

"""
X_train_unsupervised_pca, X_test_unsupervised_pca, y_train_unsupervised_pca, y_test_unsupervised_pca =train_test_split(X_train_unsupervised_pca, 
                                                                                                     y_train,
                                                                                                 test_size=0.3, 
                                                                                                 random_state=128)
"""

log_reg_with_unsupervised_TSNE = LogisticRegression(random_state=42)
log_reg_with_unsupervised_TSNE.fit(X_train_unsupervised_TSNE, y_train)
y_pred_unsupervised_TSNE = log_reg_with_unsupervised_TSNE.predict(X_test_unsupervised_TSNE)
print(classification_report(y_test, y_pred_unsupervised_TSNE ))

'''In the SVD and PCA graphs, the 2 very defined components are observed, while in TSNE they are very dispersed. The SVD presicion is 0.95, the PCA precision is 0.95 and TSNE's precision is 0.38'''