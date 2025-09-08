
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def generate_L(dim, epsilon=0.1):
    """
    Generate a random invertible matrix L of given dimension.
    """
    # Generate a random matrix L
    L = np.random.randn(dim, dim)

    # Ensure L is invertible (optional)
    L += epsilon * np.eye(dim)
    # Sigma = L @ L.T

    return L

# def generate_covariance_matrix(L)
#     Sigma = L @ L.T
#     return Sigma
# #     A = np.random.rand(D, D)
# #     return np.dot(A, A.T)\

# use to obtain our label
def top_k_eigen(covariance, k):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort the eigenvalues in descending order, and get their indices
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Select the top k eigenvalues and corresponding eigenvectors
    top_k_eigenvalues = eigenvalues[sorted_indices][:k]
    top_k_eigenvectors = eigenvectors[:, sorted_indices][:, :k]

    return top_k_eigenvalues, top_k_eigenvectors

# Function to generate a single X matrix with shape N*D from a random mean and covariance matrix
def generate_X(N,D,mean,L):
    # X = np.zeros((N, D))
    # for i in range(N):
    #   X[i, :] = np.random.multivariate_normal(mean, covariance)
    # mean = np.random.rand(D)
    Z = np.random.randn(N, D)
    # Z_i = np.random.normal(loc=0, scale=1, size=D)
    X = Z @ L.T + mean
    return X




def compute_covariance_matrix(X):

    N = X.shape[0]
    
    mean_vector = np.mean(X, axis=0)        # Shape: (D,)
    X_centered = X - mean_vector            # Shape: (N, D)
    # check x 
    cov_matrix = (1 / (N - 1)) * np.dot(X_centered.T, X_centered)
    
    return cov_matrix

def compute_covariance_matrix_scale_d(X, D):

    N = X.shape[0]
    
    mean_vector = np.mean(X, axis=0)        # Shape: (D,)
    X_centered = X - mean_vector            # Shape: (N, D)
    # check x 
    cov_matrix = (1 / (N - 1)) * np.dot(X_centered.T, X_centered) / D
    
    return cov_matrix



def generate_batch(batch_size, D, N, k, input_is_cov, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_batch = []
    Y_batch = []
    Y_vector_batch = []
  
    for _ in range(batch_size):

        # generate mean from uniform distribution, need discussion
        mean =0
        L = generate_L(D)
        X = generate_X(N, D, mean, L)
        sample_covariance = compute_covariance_matrix(X)
        if input_is_cov:
            X = sample_covariance
            print(f"Input is covariance matrix with shape {X.shape} ")
        # X = generate_X(N, D, mean, L)
        # print(f"shape of one X: {X.shape}")
        Y, Y_vector= top_k_eigen(sample_covariance, k)

        X_batch.append(X)
        Y_batch.append(Y)
        Y_vector_batch.append(Y_vector)

    X_batch = torch.tensor(np.array(X_batch), dtype=torch.float32).to(device)
    Y_batch = torch.tensor(np.array(Y_batch), dtype=torch.float32).to(device)
    Y_vector_batch = torch.tensor(np.array(Y_vector_batch), dtype=torch.float32).to(device)
    # Y_vector_batch = Y_vector_batch.view(Y_vector_batch.shape[0], -1)
    # print(f"shape of X_batch: {X_batch.shape}")
    # print(f"shape of Y_batch: {Y_batch.shape}")
    return X_batch, Y_batch, Y_vector_batch


def generate_batch_scale_d(batch_size, D, N, k, input_is_cov, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_batch = []
    Y_batch = []
    Y_vector_batch = []
  
    for _ in range(batch_size):

        # generate mean from uniform distribution, need discussion
        mean =0
        L = generate_L(D)
        X = generate_X(N, D, mean, L)
        sample_covariance = compute_covariance_matrix_scale_d(X,D)
        if input_is_cov:
            X = sample_covariance
            print(f"Input is covariance matrix with shape {X.shape} ")
        # X = generate_X(N, D, mean, L)
        # print(f"shape of one X: {X.shape}")
        Y, Y_vector= top_k_eigen(sample_covariance, k)

        X_batch.append(X)
        Y_batch.append(Y)
        Y_vector_batch.append(Y_vector)

    X_batch = torch.tensor(np.array(X_batch), dtype=torch.float32).to(device)
    Y_batch = torch.tensor(np.array(Y_batch), dtype=torch.float32).to(device)
    Y_vector_batch = torch.tensor(np.array(Y_vector_batch), dtype=torch.float32).to(device)
    # Y_vector_batch = Y_vector_batch.view(Y_vector_batch.shape[0], -1)
    # print(f"shape of X_batch: {X_batch.shape}")
    # print(f"shape of Y_batch: {Y_batch.shape}")
    return X_batch, Y_batch, Y_vector_batch




def generate_batch_cov(batch_size, D, N, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_batch = []
    Y_batch = []
  
    for _ in range(batch_size):

        # generate mean from uniform distribution, need discussion
        mean = np.random.rand(D)
        L = generate_L(D)
        # covariance = L @ L.T

        X = generate_X(N, D, mean, L)

        # X = generate_X(N, D, mean, L)
        # print(f"shape of one X: {X.shape}")
        Y= compute_covariance_matrix(X)

        X_batch.append(X)
        Y_batch.append(Y)
        # Y_vector_batch.append(Y_vector)

    X_batch = torch.tensor(np.array(X_batch), dtype=torch.float32).to(device)
    Y_batch = torch.tensor(np.array(Y_batch), dtype=torch.float32).to(device)
    # Y_vector_batch = torch.tensor(np.array(Y_vector_batch), dtype=torch.float32).to(device)
    # Y_vector_batch = Y_vector_batch.view(Y_vector_batch.shape[0], -1)
    # print(f"shape of X_batch: {X_batch.shape}")
    # print(f"shape of Y_batch: {Y_batch.shape}")
    return X_batch, Y_batch


class CovarianceDataset(Dataset):
    def __init__(self, file_name, k,predict_vector):
        # Load the data from the .npz file
        data = np.load(file_name)
        self.X_data = data['X_data']
        self.Y_data = data['Y_data']
        self.predict_vector = predict_vector
        
        if predict_vector:
            self.Y_vector_data = data['Y_vector_data'][:,:,:k]
        else:
            self.Y_vector_data = None

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        # Convert NumPy arrays to PyTorch tensors
        X = torch.tensor(self.X_data[idx], dtype=torch.float32)
        Y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        
        if self.predict_vector:
            Y_vector = torch.tensor(self.Y_vector_data[idx], dtype=torch.float32)
            return X, Y, Y_vector
        else:
            return X, Y

class CovarianceDataset_real_world(Dataset):
    def __init__(self, file_name, k, predict_vector=True, is_test=False):
        # Load the data from the .npz file
        data = np.load(file_name)
        
        if is_test:
            self.X_data = data['X_test']  # Load test data
            self.Y_data = data['Y_test']
            if predict_vector:
                self.Y_vector_data = data['Y_vector_test'][:,:,:k]  # Load test vector data
            else:
                self.Y_vector_data = None
        else:
            self.X_data = data['X_train']  # Load training data
            self.Y_data = data['Y_train']
            if predict_vector:
                self.Y_vector_data = data['Y_vector_train'][:,:,:k]  # Load training vector data
            else:
                self.Y_vector_data = None

        self.predict_vector = predict_vector

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        # Convert NumPy arrays to PyTorch tensors
        X = torch.tensor(self.X_data[idx], dtype=torch.float32)
        Y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        
        if self.predict_vector:
            Y_vector = torch.tensor(self.Y_vector_data[idx], dtype=torch.float32)
            return X, Y, Y_vector
        else:
            return X, Y





