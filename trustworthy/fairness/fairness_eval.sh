CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model apc --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model whisper_base --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model whisper_tiny --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model whisper_small --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model wav2vec2_0 --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model tera --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model wavlm --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
