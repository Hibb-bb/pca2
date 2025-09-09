import numpy as np
from tqdm import tqdm
from data_generation import compute_covariance_matrix, top_k_eigen
import os
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


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


def generate_mnist_ood(
    num_train_samples,      # How many training X-blocks to generate
    num_test_samples,       # How many testing X-blocks to generate
    D,                      # Dimension to which images are downscaled via SVD
    N,                      # How many images in each X-block
    k,                      # For top-k eigen
    input_is_cov,          # If True, X is the D x D cov matrix
    predict_vector,        # If True, also store eigenvectors
    train_digits,          # List of digits for training (e.g., [0,1,2,3,4])
    test_digits,           # List of digits for testing (e.g., [5,6,7,8,9])
    file_name              # Output .npz
):
    """
    Generate MNIST dataset for OOD experiments by splitting digits.
    
    Args:
        train_digits: List of digit classes for training (e.g., [0,1,2,3,4])
        test_digits: List of digit classes for testing (e.g., [5,6,7,8,9])
    """
    os.makedirs('dataset', exist_ok=True)

    # 1) Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    training_set = torchvision.datasets.MNIST(
        './data', train=True, transform=transform, download=True
    )
    testing_set = torchvision.datasets.MNIST(
        './data', train=False, transform=transform, download=True
    )

    # 2) Get data and labels
    X_train_full = training_set.data.view(-1, 784).numpy()
    y_train_full = training_set.targets.numpy()
    X_test_full = testing_set.data.view(-1, 784).numpy()
    y_test_full = testing_set.targets.numpy()

    print(f"Full training shape: {X_train_full.shape}")
    print(f"Full testing shape: {X_test_full.shape}")

    # 3) Filter by digit classes
    # Training data: only use specified training digits
    train_mask = np.isin(y_train_full, train_digits)
    X_train_filtered = X_train_full[train_mask]
    y_train_filtered = y_train_full[train_mask]
    
    # Testing data: only use specified testing digits
    test_mask = np.isin(y_test_full, test_digits)
    X_test_filtered = X_test_full[test_mask]
    y_test_filtered = y_test_full[test_mask]

    print(f"Training digits: {train_digits}")
    print(f"Testing digits: {test_digits}")
    print(f"Filtered training shape: {X_train_filtered.shape}")
    print(f"Filtered testing shape: {X_test_filtered.shape}")
    print(f"Training digit distribution: {np.bincount(y_train_filtered)}")
    print(f"Testing digit distribution: {np.bincount(y_test_filtered)}")

    # 4) Center data
    X_centered_train = X_train_filtered - np.mean(X_train_filtered, axis=0)
    X_centered_test = X_test_filtered - np.mean(X_test_filtered, axis=0)

    # 5) SVD for dimensionality reduction
    U_train, S_train, Vt_train = np.linalg.svd(X_centered_train, full_matrices=False)
    U_test, S_test, Vt_test = np.linalg.svd(X_centered_test, full_matrices=False)

    # 6) Project onto top D principal components
    X_reduced_train = np.dot(X_centered_train, Vt_train.T[:, :D])
    X_reduced_test = np.dot(X_centered_test, Vt_test.T[:, :D])
    
    # 7) Standardize
    scaler = StandardScaler()
    X_reduced_train = scaler.fit_transform(X_reduced_train)
    X_reduced_test = scaler.transform(X_reduced_test)

    print(f"X_reduced_train shape: {X_reduced_train.shape}")
    print(f"X_reduced_test shape: {X_reduced_test.shape}")

    # 8) Generate training blocks in parallel
    print("Generating training blocks...")
    train_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_block)(
            X_reduced_train, N, k, input_is_cov, predict_vector
        ) for _ in range(num_train_samples)
    )

    # 9) Generate testing blocks in parallel
    print("Generating testing blocks...")
    test_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_block)(
            X_reduced_test, N, k, input_is_cov, predict_vector
        ) for _ in range(num_test_samples)
    )

    # 10) Unpack the results
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

    # 11) Convert to NumPy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    X_data_test = np.array(X_data_test)
    Y_data_test = np.array(Y_data_test)

    print("X_data shape:", X_data.shape)
    print("Y_data shape:", Y_data.shape)
    print("X_data_test shape:", X_data_test.shape)
    print("Y_data_test shape:", Y_data_test.shape)

    if predict_vector:
        Y_vector_data = np.array(Y_vector_data)
        Y_vector_data_test = np.array(Y_vector_data_test)
        print("Y_vector_data shape:", Y_vector_data.shape)
        print("Y_vector_data_test shape:", Y_vector_data_test.shape)

    # 12) Save data to .npz file
    save_data = {
        'X_train': X_data,
        'Y_train': Y_data,
        'X_test': X_data_test,
        'Y_test': Y_data_test,
        'train_digits': train_digits,
        'test_digits': test_digits
    }
    
    if predict_vector:
        save_data.update({
            'Y_vector_train': Y_vector_data,
            'Y_vector_test': Y_vector_data_test
        })

    np.savez(file_name, **save_data)
    print(f"OOD dataset saved to {file_name}")


if __name__ == "__main__":
    # Example: Train on digits 0-4, test on digits 5-9

    # for n in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    generate_mnist_ood(
        num_train_samples=6400000,   # Reduce for testing
        num_test_samples=10000,
        D=10,
        N=10,
        k=5,
        input_is_cov=False,
        predict_vector=True,
        train_digits=[0, 1, 2, 3, 4],
        test_digits=[5, 6, 7, 8, 9],
        file_name="dataset/mnist_ood_digits_0to4_train_5to9_test.npz"
    )
    
    # You can also try other splits:
    # Even vs Odd digits
    # generate_mnist_ood(
    #     num_train_samples=1000000,
    #     num_test_samples=2048,
    #     D=10, N=10, k=5,
    #     input_is_cov=False, predict_vector=True,
    #     train_digits=[0, 2, 4, 6, 8],  # Even digits
    #     test_digits=[1, 3, 5, 7, 9],   # Odd digits
    #     file_name="dataset/mnist_ood_even_train_odd_test.npz"
    # )
