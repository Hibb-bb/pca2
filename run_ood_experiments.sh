#!/bin/bash

export WANDB_API_KEY=

# # First, generate the OOD datasets
# echo "Generating OOD datasets..."

# # Dataset 1: Train on digits 0-4, test on digits 5-9
# python3 -c "
# from data_prepare_ood import generate_mnist_ood
# generate_mnist_ood(
#     num_train_samples=9600000,
#     num_test_samples=2048, 
#     D=10, N=10, k=5,
#     input_is_cov=False, predict_vector=True,
#     train_digits=[0, 1, 2, 3, 4],
#     test_digits=[5, 6, 7, 8, 9],
#     file_name='dataset/mnist_ood_digits_0to4_train_5to9_test.npz'
# )
# "

# # Dataset 2: Even vs Odd digits
# python3 -c "
# from data_prepare_ood import generate_mnist_ood
# generate_mnist_ood(
#     num_train_samples=9600000,
#     num_test_samples=2048,
#     D=10, N=10, k=5,
#     input_is_cov=False, predict_vector=True,
#     train_digits=[0, 2, 4, 6, 8],  # Even digits
#     test_digits=[1, 3, 5, 7, 9],   # Odd digits
#     file_name='dataset/mnist_ood_even_train_odd_test.npz'
# )
# "

# echo "OOD datasets generated. Starting training experiments..."

# Experiment 1: Train on digits 0-4, test on digits 5-9
for seed in 1234 1235 1236 1237 1238
do
  for k in 3 4 5
  do
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        echo "Training on digits 0-4, testing on 5-9, seed=${seed}, k=${k}"
        python3 main_realworld_0in1.py \
          --batch_size 64 \
          --n_layer 12 \
          --n_embd 256 \
          --n_head 8 \
          --D 10 \
          --N 10 \
          --n_training_data 9600000 \
          --k $k \
          --predict_vector \
          --is_relu \
          --is_layernorm \
          --dataset dataset/mnist_ood_digits_0to4_train_5to9_test.npz \
          --run_name ood_mnist_0to4_train_5to9_test_seed_${seed}_k_${k} \
          --save_model_to ckpt/ood_mnist_0to4_train_5to9_test_seed_${seed}_k_${k}_ratio${ratio} \
          --seed $seed \
          --lr 0.0005 \
          --eval_step 5000 \
          --ratio_use $ratio \
          --csv_name ckpt/ood_mnist_0to4_train_5to9_test_seed_${seed}_k_${k}_ratio${ratio}.csv \
          --train
    done
  done
done

# # Experiment 2: Even vs Odd digits
# for seed in 1234 1235 1236
# do
#   for k in 3 4 5
#   do
#     echo "Training on even digits, testing on odd digits, seed=${seed}, k=${k}"
#     python3 main_realworld_0in1.py \
#       --batch_size 64 \
#       --n_layer 12 \
#       --n_embd 256 \
#       --n_head 8 \
#       --D 10 \
#       --N 10 \
#       --n_training_data 9600000 \
#       --k $k \
#       --predict_vector \
#       --is_relu \
#       --is_layernorm \
#       --dataset dataset/mnist_ood_even_train_odd_test.npz \
#       --run_name ood_mnist_even_train_odd_test_seed_${seed}_k_${k} \
#       --save_model_to ckpt/ood_mnist_even_train_odd_test_seed_${seed}_k_${k} \
#       --seed $seed \
#       --lr 0.0005 \
#       --eval_step 5000 \
#       --train
#   done
# done

echo "OOD experiments completed! bish"
