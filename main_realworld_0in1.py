import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import time
import wandb
from sklearn.datasets import make_spd_matrix
from random import randint
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import matplotlib.pyplot as plt
import csv
import os

from data_generation import generate_L, top_k_eigen, generate_X, generate_batch, generate_batch_cov, CovarianceDataset, generate_batch_scale_d, CovarianceDataset_real_world
from model import TransformerModel, TransformerModel_drop
from loss import MeanRelativeSquaredError

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer-based PCA")

    parser.add_argument("--D", type=int, default=2, help="Dimension of each column vector")
    parser.add_argument("--N", type=int, default=5, help="Number of columns in each X matrix")
    parser.add_argument("--k", type=int, default=1, help="Number of top eigenvalues to use as labels")
    parser.add_argument("--n_embd", type=int, default=64, help="Embedding size for the transformer")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers in the transformer")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads in the transformer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--training_steps", type=int, default=60000, help="Total number of training steps")
    parser.add_argument("--n_training_data", type=int, default=1024000, help="Total number of training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_step", type=int, default=50,help="log the loss")
    parser.add_argument("--seed", type=int, default=1234,help="log the loss")
    parser.add_argument("--plot_name", type=str, help="Specify the name of the output plot file.")
    parser.add_argument("--csv_name", type=str, help="Specify the name of the output csv file.")
    parser.add_argument("--dataset", type=str, help="Specify the name of the output csv file.")
    parser.add_argument("--run_name", type=str, help="Specify the name of the output csv file.")
    parser.add_argument("--save_model_to", type=str, help="Specify the name of the output csv file.")
    parser.add_argument("--resume", type=str, help="Specify the checkpoint path.")
    parser.add_argument("--input_is_cov", action='store_true', help="Flag to specify if input is a covariance matrix")
    parser.add_argument("--predict_vector", action='store_true', help="Flag to specify if you want to predict eigenvectors") 
    parser.add_argument("--predict_cov", action='store_true', help="Flag to specify if you want to predict covariance matrix") 
    parser.add_argument("--is_relu", action='store_true', help="Flag to specify if you want relu in attention")
    parser.add_argument("--is_layernorm", action='store_true', help="Flag to specify if you want layer normalization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--fine_tune_from",type=str, help="Number of epochs for training")
    parser.add_argument("--train",action='store_true', help="Number of epochs for training")
    parser.add_argument("--test",action='store_true', help="Number of epochs for training")
    parser.add_argument("--penalty",action='store_true', help="Number of epochs for training")
    parser.add_argument("--eigenspace",action='store_true', help="Number of epochs for training")
    parser.add_argument("--eval_step",type=int, default=5000,help="Number of epochs for training")


  
    return parser.parse_args()

def save_checkpoint(model, optimizer, step, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step




if __name__ == "__main__":
    torch.set_num_threads(6)
    args = parse_args()
  
    # wandb.init(project="transformer_pca", config={
    #     "learning_rate": args.lr,
    #     "batch_size": args.batch_size,
    #     "architecture": "gpt2",
    #     "dataset": "Generated Covariance Data",
    #     "training_steps": args.training_steps,
    #     "D": args.D,
    #     "N": args.N,
    #     "top_k_eigenvalues": args.k,
    #     "n_embd": args.n_embd,
    #     "n_layer": args.n_layer,
    #     "n_head": args.n_head,
    #     "input_is_covariance": args.input_is_cov,
    #     "predict_vector": args.predict_vector
    # })


    
    # Parameters
    D = args.D  # Dimension of each column vector
    # print("D:" D)
    N = args.N  # Number of columns in each X matrix
    k = args.k  # Number of top eigenvalues to use as labels
    n_embd = args.n_embd
    n_layer = args.n_layer
    n_head = args.n_head
    n_training_data = args.n_training_data
    # print(n_training_data)
    
    # training parameters
    input_is_cov = args.input_is_cov # True if input of transformer is covariance matrix, thus we can test whether transformer can do power iteration method
    print("input_is_cov: ",input_is_cov)
    predict_vector = args.predict_vector
    print("predict_vector: ",predict_vector)
    predict_cov = args.predict_cov
    print("predict_cov: ",predict_cov)
    batch_size = args.batch_size
    csv_file = args.csv_name
    plot_file = args.plot_name
    # training_steps = args.training_steps
    print_every = args.log_step
    lr = args.lr
    is_relu = args.is_relu
    is_layernorm = args.is_layernorm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    training_steps = int(n_training_data / batch_size)
    wandb.init(project="transformer_pca", name=args.run_name,config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "architecture": "gpt2",
    "dataset": "Generated Covariance Data",
    "training_steps": training_steps,
    "D": D,
    "N": N,
    "top_k_eigenvalues": k,
    "n_embd": n_embd,
    "n_layer": n_layer,
    "n_head": n_head,
    "input_is_covariance": input_is_cov,
    "predict_vector": predict_vector,
    "predict_cov":predict_cov,
    "is_relu":is_relu,
    "is_layernorm": is_layernorm,
})
  

    data_test = np.load(args.dataset)
    X_test = data_test['X_test']
    Y_test = data_test['Y_test']
    if 'Y_vector_train' in data_test:
        Y_vector_test = data_test['Y_vector_test']
        print("Y_vector_traom shape: ", Y_vector_test.shape)
    else:
        Y_vector_test = None
    print("X_test shape: ", X_test.shape)
    print("Y_test shape: ", Y_test.shape)
    
    if Y_vector_test is not None:
        print("Y_vector_test shape: ", Y_vector_test.shape)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
    Y_vector_test  = torch.tensor(Y_vector_test, dtype=torch.float32).to(device)
    Y_vector_test = Y_vector_test[:,:, :k]
    print("Y_vector_test shape: ", Y_vector_test.shape)
    Y_vector_test = torch.transpose(Y_vector_test, 1, 2)
    print("Y_vector_test shape: ", Y_vector_test.shape)  



    if args.train:
        # define modelm, optimizer, and loss
        model = TransformerModel_drop(D, N, N+10, n_embd=n_embd, n_layer=n_layer, n_head=n_head,input_is_cov=input_is_cov, predict_vector=predict_vector, predict_cov = predict_cov, is_relu = is_relu, is_layernorm=is_layernorm, k=k).to(device)
        print(f"model architecture:{model.name}")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = MeanRelativeSquaredError()
        # scheduler = ExponentialLR(optimizer, gamma=0.9)

        if args.resume:
            print("Resuming training from checkpoint")
            checkpoint_path = args.resume
            if os.path.exists(checkpoint_path):
                model, optimizer, start_step = load_checkpoint(checkpoint_path, model, optimizer)
                print(f"Resuming training from step {start_step}")
            else:
                raise ValueError("No checkpoint found")
        else:
            print("Starting training from scratch")

        train_losses = []
        steps = []
        # Training loop with validation every 1000 steps
        start_time = time.time()
        loss_sum = 0

        # Check if the file exists to write headers, otherwise create a new file with headers
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(['Step', 'Training Loss', 'Elapsed Time'])


        dataset = CovarianceDataset_real_world(args.dataset, k=k,predict_vector=True)
        # dataset_2 = CovarianceDataset("dataset/multivariate_gaussian_dataset_D_30_2560000.npz", k=k,predict_vector=True)
        # Define DataLoader for batch processing
        # combined_dataset = ConcatDataset([dataset, dataset_2])
        torch.manual_seed(args.seed)
        # dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)



        training_steps = int(n_training_data / batch_size)
        print("training_steps", training_steps)
        print_every = int(1024 / batch_size)
        if not args.resume:
            start_step = 0
        # start_step = 0
        for step, (X_train_batch, Y_train_batch, *Y_vector_batch) in enumerate(dataloader, start=start_step):
            start_step = step
            model.train()
            if step == 0:
                print(f"shape of Y_train_batch: {Y_train_batch.shape}")

            X_train_batch = X_train_batch.to(device)
            Y_train_batch = Y_train_batch[:,:k].to(device)
            if predict_vector and len(Y_vector_batch) > 0:
                Y_vector_batch = Y_vector_batch[0].to(device)  # Move eigenvectors to GPU
                # print(f"shape of Y_vector_batch in train loop before transpose: {Y_vector_batch.shape}")
                Y_vector_batch = torch.transpose(Y_vector_batch, 1,2).to(device) # Transpose eigenvectors to (batch_size, k, D)
            # print(f"shape of Y_vector_batch: {Y_vector_batch.shape}")
            if step == 0:
                print(f"shape of X_train_batch: {X_train_batch.shape}")
                print(f"shape of Y_train_batch: {Y_train_batch.shape}")
                if predict_vector:
                    print(f"shape of Y_vector_batch: {Y_vector_batch.shape}")
            
            output = model(X_train_batch)
            if step == 0:
                print("shape of output", output.shape)
            # print(f"shape of output: {output.shape}")
            # if predict_cov:
            #     output = output[:, :D, :D]
            #     # print("shape of output", output.shape)
            if predict_vector:
                # print(f"shape of X_train_batch: {X_train_batch.shape}")
                # print("shape of output", output.shape)
                output = output.view(output.shape[0], k, D) # Reshape output to (batch_size, k, D)
                if step == 0:
                    print(f"shape of output: {output.shape}")
                if args.penalty:
                    weights = torch.arange(1, k + 1, device=output.device).float()  # Weights: [1, 2, ..., k]
                    # weights = weights / weights.sum()  # Normalize weights to sum to 1

                    # Compute cosine similarity for each eigenvector (k dimension)
                    cosine_similarities = F.cosine_similarity(output, Y_vector_batch, dim=-1)  # Shape: (batch_size, k)
                    
                    # Apply weights to penalize higher k values
                    weighted_loss = (1 - cosine_similarities) * weights  # Shape: (batch_size, k)

                    # Take the mean across batch and k dimensions
                    loss = weighted_loss.mean()
                elif args.eigenspace:
                    pred = output.transpose(1,2)        # (batch_size, D, k)
                    true = Y_vector_batch.transpose(1,2)# (batch_size, D, k)

                    # 2) Form projectors P_pred = E E^T, P_true = F F^T
                    #    which become (batch_size, D, D) matrices:
                    P_pred = torch.bmm(pred, pred.transpose(1,2))  # E * E^T
                    P_true = torch.bmm(true, true.transpose(1,2))  # F * F^T

                    # 3) Compute the difference and its Frobenius norm
                    diff = P_pred - P_true
                    fro_sq = diff.pow(2).sum(dim=(1,2))  # sum of squares over last two dims

                    # 4) Finally, the subspace “distance” loss is 1/2 ||EE^T - FF^T||^2_F
                    #    You can mean over the batch if you wish:
                    loss = 0.5 * fro_sq.mean()
                else:
                    loss = 1 - F.cosine_similarity(output, Y_vector_batch, dim=-1).mean()


            
                # loss = 1 - F.cosine_similarity(output, Y_vector_batch).mean()
            else:
                loss = criterion(output, Y_train_batch)

            # epoch_loss_sum += loss.item()
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Print and log progress every 'print_every' steps

            if (step+1) % print_every == 0:
                elapsed_time = time.time() - start_time
                avg_loss = loss_sum / print_every
                train_losses.append(avg_loss)
                steps.append(step + 1)
                print()
                print(f"Step [{step+1}/{training_steps}], Training Loss: {avg_loss:.6f}, Elapsed Time: {elapsed_time:.2f}s")
                print()
                wandb.log({"training_loss": avg_loss, "step": step+1})
                if predict_vector:
                    # pass
                    if (step+1) % print_every*10 == 0:
                        print(f"True top-{k} eigenvectors for 0th data in a batch:{Y_vector_batch[0,:]}")
                        print(f"Predicted top-{k} eigenvectors for 0th data in a batch:{output[0,:]}")
                #   print(f"True top-{k} eigenvectors for 0th data in a batch:{Y_vector_batch[0,:]}")
                #   print(f"Predicted top-{k} eigenvectors for 0th data in a batch:{output[0,:]}")
                # elif predict_cov:
                #   print(f"True covariance matrix for first data in first batch:{Y_train_batch[0,:]}")
                #   print(f"Predicted covariance matrix for first data in first batch:{output[0,:]}")
                # else:
                #   print(f"True top-{k} eigenvalues for 0th~5th data in a batch:{Y_train_batch[:1,:k]}")
                #   print(f"Predicted top-{k} eigenvalues for 0th~5th data in a batch:{output[:1,:k]}")

                    
                start_time = time.time() 
                loss_sum = 0

                # start_step = step + 1

            # Evaluate the model 
            if (step + 1) % args.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    eval_output = model(X_test)
                    if predict_vector:
                        eval_output = eval_output.view(X_test.shape[0], k, D)
                        if args.eigenspace:
                            pred = eval_output.transpose(1,2)        # (batch_size, D, k)
                            true = Y_vector_test.transpose(1,2)# (batch_size, D, k)

                            # 2) Form projectors P_pred = E E^T, P_true = F F^T
                            #    which become (batch_size, D, D) matrices:
                            P_pred = torch.bmm(pred, pred.transpose(1,2))  # E * E^T
                            P_true = torch.bmm(true, true.transpose(1,2))  # F * F^T

                            # 3) Compute the difference and its Frobenius norm
                            diff = P_pred - P_true
                            fro_sq = diff.pow(2).sum(dim=(1,2))  # sum of squares over last two dims
                            # 4) Finally, the subspace “distance” loss is 1/2 ||EE^T - FF^T||^2_F
                            #    You can mean over the batch if you wish:
                            loss = 0.5 * fro_sq.mean()
                            wandb.log({"Distance of eigenspace": loss, "step": step + 1})
                        else:
                            testing_errors = []
                            for i in range(k):
                                print("Y_vector_test shape: ", Y_vector_test.shape)
                                error = F.cosine_similarity(eval_output[:, i, :], Y_vector_test[:, i, :], dim=-1).mean()
                                testing_errors.append(error.item())
                                wandb.log({f"eigenvector_cos_similarity_{i+1}": error, "step": step + 1})
                            print(f"Epoch {step+1} Testing Cosine Similarity Errors: {testing_errors}")
                    else:
                        testing_errors = []
                        for i in range(k):
                            error = criterion(eval_output[:, i], Y_test[:, i])
                            testing_errors.append(error.item())
                            wandb.log({f"eigenvalue_error_{i+1}_step_{step+1}": error})
                        print(f"Epoch {step+1} Testing Errors: {testing_errors}")
        
            # if (step + 1) % 50000 == 0:
            #     # path = f"{args.save_model_to}_step_{step + 1}"
            #     save_checkpoint(model, optimizer, start_step)

        # avg_loss_epoch = epoch_loss_sum / len(dataloader)  # Average loss for the entire epoch
        # print(f"Epoch [{epoch+1}/{args.epochs}], Average Epoch Loss: {avg_loss_epoch:.6f}")
        # wandb.log({"average_epoch_loss": avg_loss_epoch, "epoch": epoch + 1})
        # # torch.save(model.state_dict(), args.save_model_to)
        save_checkpoint(model, optimizer, start_step, args.save_model_to)


    if args.test:
        model_eval = TransformerModel(D, N, N+10, n_embd=n_embd, n_layer=n_layer, n_head=n_head,input_is_cov=input_is_cov, predict_vector=predict_vector, predict_cov = predict_cov, is_relu = is_relu, is_layernorm=is_layernorm, k=k).to(device)
        checkpoint = torch.load(args.save_model_to,weights_only=True)
        model_eval.load_state_dict(checkpoint['model_state_dict'])
        print("evaluation start.")

        # data_test = np.load(args.dataset)
        # X_test = data_test['X_test']
        # Y_test = data_test['Y_test']
        

        model_eval.eval()
        
        if predict_vector:
            # Y_vector_test = torch.transpose(Y_vector_test, 1, 2)
            print("Y_vector_shape: ", Y_vector_test.shape)
            eval_output = model_eval(X_test)
            eval_output = eval_output.view(X_test.shape[0], k, D)
            print(f"eval_output shape: {eval_output.shape}")
            testing_errors = []
            for i in range(k):
                error = F.cosine_similarity(eval_output[:,i,:], Y_vector_test[:,i,:],dim=-1).mean()
                testing_errors.append(error.item())
                wandb.log({f"eigenvector_cos_similarity_{i+1}": error})
        else:
            eval_output = model_eval(X_test)
            print(f"shape of eval_output: {eval_output.shape}")
            print(f"shape of Y_test_1280: {Y_test.shape}")
            testing_errors = []

            for i in range(k):
                error = criterion(eval_output[:, i], Y_test[:, i])
                testing_errors.append(error.item())
                wandb.log({f"{i+1}-eigenvalue_testing_error": error})

        csv_filename = "csv/train_to_60k_divide_80_fashion_mnist_eigenvector.csv"

        # Check if the file exists
        if not os.path.exists(csv_filename):
            # Create and write headers if the file doesn't exist
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                headers = ["Eigenvector Index"] + [f"Run {j+1}" for j in range(len(testing_errors))]
                writer.writerow(headers)

        # Append the current run's errors to the CSV file
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            row = [f"Eigenvector {i+1}" for i in range(k)]
            writer.writerow(testing_errors)

        print(f"cos_similarity written to {csv_filename}")

  
  
  
