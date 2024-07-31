import numpy as np

def create_matrices(n, a, b):
    A = np.eye(n, dtype = int) # Creates a 2D array with 1's in the diagnal
    B = np.eye(n, dtype = int) # An identity matrix
    for i in range (n - 1):
        A[i, i + 1] = a # Modifies matrix a such that diagnal above the 1's is equal to a
        B[i + 1, i] = b # Modifies matrix a such that diagnal below the 1's is equal to b
    return np.linalg.matrix_power(A, a), np.linalg.matrix_power(B, b) #Returns a tuple of A^a and B^a
