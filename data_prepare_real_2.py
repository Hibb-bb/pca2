import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from data_generation import generate_L, top_k_eigen, generate_X, compute_covariance_matrix, compute_covariance_matrix_scale_d
import os
from sklearn.datasets import fetch_openml
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler



import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from joblib import Parallel, delayed

# def compute_covariance_matrix(X):
#     X_centered = X - X.mean(axis=0, keepdims=True)
#     return (X_centered.T @ X_centered) / (X.shape[0] - 1)

# def top_k_eigen(cov_matrix, k):
#     eigvals, eigvecs = np.linalg.eig(cov_matrix)
#     idx = np.argsort(eigvals)[::-1]
#     return eigvals[idx[:k]], eigvecs[:, idx[:k]]

def _process_one_block(X_reduced, N, k, input_is_cov, predict_vector):
    """
    Helper function to sample one block, compute covariance, eigenvalues/vectors.
    This function will be executed in parallel via joblib.
    """
    random_idx = np.random.choice(X_reduced.shape[0], N, replace=False)
    one_input = X_reduced[random_idx, :]  # shape (N, D)

    if input_is_cov:
        X_block = compute_covariance_matrix(one_input)
    else:
        X_block = one_input

    cov_for_label = compute_covariance_matrix(one_input)
    Y, Y_vec = top_k_eigen(cov_for_label, k)

    if predict_vector:
        return X_block, Y, Y_vec
    else:
        return X_block, Y, None

def generate_fashion_mnist(
    num_train_samples,  # How many training X-blocks to generate
    num_test_samples,   # How many testing X-blocks to generate
    D,                  # Dimension to which images are downscaled via SVD (top D principal components)
    N,                  # How many images in each X-block
    k,                  # For top-k eigen
    input_is_cov,       # If True, X is the D x D cov matrix
    predict_vector,     # If True, also store eigenvectors
    file_name           # Output .npz
):
    os.makedirs('dataset', exist_ok=True)

    # 1) Load FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # normalized single channel if needed
    ])
    training_set = torchvision.datasets.FashionMNIST(
        './data', train=True, transform=transform, download=True
    )
    testing_set = torchvision.datasets.FashionMNIST(
        './data', train=False, transform=transform, download=True
    )

    # 2) Flatten each image (28x28 -> 784); the transform’s normalization is in torch-space
    X_train_full = training_set.data.view(-1, 784).numpy()
    X_test_full = testing_set.data.view(-1, 784).numpy()

    print("Full training shape:", X_train_full.shape)  # (60000, 784)
    print("Full testing shape:", X_test_full.shape)    # (10000, 784)

    # 3) Center data
    X_centered_train = X_train_full - np.mean(X_train_full, axis=0)
    X_centered_test = X_test_full - np.mean(X_test_full, axis=0)

    # 4) SVD for dimensionality reduction
    U_train, S_train, Vt_train = np.linalg.svd(X_centered_train, full_matrices=False)
    U_test, S_test, Vt_test = np.linalg.svd(X_centered_test, full_matrices=False)

    # 5) Project onto top D principal components
    X_reduced_train = np.dot(X_centered_train, Vt_train.T[:, :D])
    X_reduced_test = np.dot(X_centered_test, Vt_test.T[:, :D])
    scaler = StandardScaler()
    X_reduced_train = scaler.fit_transform(X_reduced_train)
    X_reduced_test = scaler.transform(X_reduced_test)

    print("X_reduced_train shape:", X_reduced_train.shape)  # (60000, D)
    print("X_reduced_test shape:", X_reduced_test.shape)    # (10000, D)

    # 6) Generate training blocks in parallel
    train_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_block)(
            X_reduced_train, N, k, input_is_cov, predict_vector
        ) for _ in range(num_train_samples)
    )

    # 7) Generate testing blocks in parallel
    test_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_block)(
            X_reduced_test, N, k, input_is_cov, predict_vector
        ) for _ in range(num_test_samples)
    )

    # 8) Unpack the results
    X_data = []
    Y_data = []
    Y_vector_data = []
    for X_block, Y, Y_vec in train_results:
        X_data.append(X_block)
        Y_data.append(Y)
        if predict_vector:
            Y_vector_data.append(Y_vec)

    X_data_test = []
    Y_data_test = []
    Y_vector_data_test = []
    for X_block, Y, Y_vec in test_results:
        X_data_test.append(X_block)
        Y_data_test.append(Y)
        if predict_vector:
            Y_vector_data_test.append(Y_vec)

    # 9) Convert to NumPy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    X_data_test = np.array(X_data_test)
    Y_data_test = np.array(Y_data_test)

    print("X_data shape:", X_data.shape)       # (num_train_samples, N, D) or (num_train_samples, D, D)
    print("Y_data shape:", Y_data.shape)       # (num_train_samples, k)
    print("X_data_test shape:", X_data_test.shape)
    print("Y_data_test shape:", Y_data_test.shape)

    if predict_vector:
        Y_vector_data = np.array(Y_vector_data)
        Y_vector_data_test = np.array(Y_vector_data_test)
        print("Y_vector_data shape:", Y_vector_data.shape)         # (num_train_samples, D, k) if input_is_cov=False
        print("Y_vector_data_test shape:", Y_vector_data_test.shape)

    # 10) Save data to .npz file
    if predict_vector:
        np.savez(
            file_name,
            X_train=X_data,
            Y_train=Y_data,
            Y_vector_train=Y_vector_data,
            X_test=X_data_test,
            Y_test=Y_data_test,
            Y_vector_test=Y_vector_data_test,
        )
    else:
        np.savez(
            file_name,
            X_train=X_data,
            Y_train=Y_data,
            X_test=X_data_test,
            Y_test=Y_data_test,
        )

    print(f"Dataset saved to {file_name}")



def generate_mnist(
    num_train_samples,  # How many training X-blocks to generate
    num_test_samples,   # How many testing X-blocks to generate
    D,                  # Dimension to which images are downscaled via SVD (top D principal components)
    N,                  # How many images in each X-block
    k,                  # For top-k eigen
    input_is_cov,       # If True, X is the D x D cov matrix
    predict_vector,     # If True, also store eigenvectors
    file_name           # Output .npz
):
    os.makedirs('dataset', exist_ok=True)

    # 1) Load FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # normalized single channel if needed
    ])
    training_set = torchvision.datasets.MNIST(
        './data', train=True, transform=transform, download=True
    )
    testing_set = torchvision.datasets.MNIST(
        './data', train=False, transform=transform, download=True
    )

    # 2) Flatten each image (28x28 -> 784); the transform’s normalization is in torch-space
    X_train_full = training_set.data.view(-1, 784).numpy()
    X_test_full = testing_set.data.view(-1, 784).numpy()

    print("Full training shape:", X_train_full.shape)  # (60000, 784)
    print("Full testing shape:", X_test_full.shape)    # (10000, 784)

    # 3) Center data
    X_centered_train = X_train_full - np.mean(X_train_full, axis=0)
    X_centered_test = X_test_full - np.mean(X_test_full, axis=0)

    # 4) SVD for dimensionality reduction
    U_train, S_train, Vt_train = np.linalg.svd(X_centered_train, full_matrices=False)
    U_test, S_test, Vt_test = np.linalg.svd(X_centered_test, full_matrices=False)

    X_reduced_train = np.dot(X_centered_train, Vt_train.T[:, :D])
    X_reduced_test = np.dot(X_centered_test, Vt_test.T[:, :D])
    scaler = StandardScaler()
    X_reduced_train = scaler.fit_transform(X_reduced_train)
    X_reduced_test = scaler.transform(X_reduced_test)
    print("X_reduced_train shape:", X_reduced_train.shape)  # (60000, D)
    print("X_reduced_test shape:", X_reduced_test.shape)    # (10000, D)

    # 6) Generate training blocks in parallel
    train_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_block)(
            X_reduced_train, N, k, input_is_cov, predict_vector
        ) for _ in range(num_train_samples)
    )

    # 7) Generate testing blocks in parallel
    test_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_block)(
            X_reduced_test, N, k, input_is_cov, predict_vector
        ) for _ in range(num_test_samples)
    )

    # 8) Unpack the results
    X_data = []
    Y_data = []
    Y_vector_data = []
    for X_block, Y, Y_vec in train_results:
        X_data.append(X_block)
        Y_data.append(Y)
        if predict_vector:
            Y_vector_data.append(Y_vec)

    X_data_test = []
    Y_data_test = []
    Y_vector_data_test = []
    for X_block, Y, Y_vec in test_results:
        X_data_test.append(X_block)
        Y_data_test.append(Y)
        if predict_vector:
            Y_vector_data_test.append(Y_vec)

    # 9) Convert to NumPy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    X_data_test = np.array(X_data_test)
    Y_data_test = np.array(Y_data_test)

    print("X_data shape:", X_data.shape)       # (num_train_samples, N, D) or (num_train_samples, D, D)
    print("Y_data shape:", Y_data.shape)       # (num_train_samples, k)
    print("X_data_test shape:", X_data_test.shape)
    print("Y_data_test shape:", Y_data_test.shape)

    if predict_vector:
        Y_vector_data = np.array(Y_vector_data)
        Y_vector_data_test = np.array(Y_vector_data_test)
        print("Y_vector_data shape:", Y_vector_data.shape)         # (num_train_samples, D, k) if input_is_cov=False
        print("Y_vector_data_test shape:", Y_vector_data_test.shape)

    # 10) Save data to .npz file
    if predict_vector:
        np.savez(
            file_name,
            X_train=X_data,
            Y_train=Y_data,
            Y_vector_train=Y_vector_data,
            X_test=X_data_test,
            Y_test=Y_data_test,
            Y_vector_test=Y_vector_data_test,
        )
    else:
        np.savez(
            file_name,
            X_train=X_data,
            Y_train=Y_data,
            X_test=X_data_test,
            Y_test=Y_data_test,
        )

    print(f"Dataset saved to {file_name}")


if __name__ == "__main__":

    # generate_fashion_mnist(
    #     num_train_samples=9600000,
    #     num_test_samples=2048,
    #     D=10,
    #     N=10,
    #     k=5,
    #     input_is_cov=False,
    #     predict_vector=True,
    #     file_name="dataset/fresh_fashion_mnist_150k.npz"
    # )

    generate_mnist(
        num_train_samples=9600000,
        num_test_samples=2048,
        D=10,
        N=10,
        k=5,
        input_is_cov=False,
        predict_vector=True,
        file_name="dataset/fresh_mnist_150k.npz"
    )
