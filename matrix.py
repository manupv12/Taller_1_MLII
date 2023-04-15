import random
import numpy as np


class Matrix:
    def __init__(self, rows=None, columns=None):
        self.rows = rows or random.randint(1, 1000)
        self.columns = columns or random.randint(20, 100)
        self.A = np.random.rand(self.rows, self.columns)

    def trace(self):
        return np.trace(self.A)

    def rank(self):
        return np.linalg.matrix_rank(self.A)

    def determinant(self):
        if self.rows != self.columns:
            return "La matriz no es cuadrada, no se puede calcular su determinante"
        else:
            return np.linalg.det(self.A)

    def invert(self):
        if self.rows == self.columns:
            return np.linalg.inv(self.A)
        else:
            return "La matriz no es cuadrada y no puede ser invertida"

    def value_vectors_eigen(self):
        AA = self.A.T @ self.A
        AAT = self.A @ self.A.T
        eigvals_AA, eigvecs_AA = np.linalg.eig(AA)
        eigvals_AAT, eigvecs_AAT = np.linalg.eig(AAT)
        return (eigvals_AA, eigvecs_AA), (eigvals_AAT, eigvecs_AAT)

matrix = Matrix()
results={
    "La matriz A es": [matrix.rows,matrix.columns],
    "La traza de A es":matrix.trace(),
    "El rango de A es": matrix.rank(),
    "El determinate de A es": matrix.determinant(),
    "La inversa de A es": matrix.invert(),
    "Los valores y vectores propios de ATA son": matrix.value_vectors_eigen()[0],
    "Los valores y vectores propios de AAT son": matrix.value_vectors_eigen()[1]
    }

print(results)