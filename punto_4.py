import cv2
import numpy as np
from numpy.linalg import eigh, norm
from sklearn import metrics
import matplotlib.pyplot as plt
from unsupervised import SVD

def SVD_unsupervised_module(A):
    svd_p4 = SVD()
    U, s, Vt = svd_p4.fit(A)
    return U, s, Vt 

# Cargar la imagen y convertirla en una matriz de numpy
img_path = "pictureFace\self_picture.jpeg"
my_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def svd_reconstruction(image, num_singular_values):
    U, S, Vt = SVD_unsupervised_module(image)
    S_reconstructed = np.diag(S)
    S_reconstructed[num_singular_values:] = 0
    reconstructed_A = U.dot(S_reconstructed).dot(Vt)
    reconstructed_A = np.clip(reconstructed_A, 0, 255)
    reconstructed_image = reconstructed_A.astype(np.uint8)
    return reconstructed_image

num_singular_values_list = [10, 50, 100, 256]
num_images = len(num_singular_values_list)

fig, axs = plt.subplots(1, num_images, figsize=(15, 15))

for i in range(num_images):
    reconstructed_image = svd_reconstruction(my_image, num_singular_values_list[i])
    axs[i].imshow(reconstructed_image, cmap='gray')
    axs[i].set_title(f"{num_singular_values_list[i]} Singular Values")
    axs[i].axis('off')

plt.show()

'''To calculate how different my image is from the approximations, the distance can be calculated in Euclidean norm, after linearizing each of the matrices, as in point 2.'''