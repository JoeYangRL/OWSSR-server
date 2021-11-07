## CIFAR100 B0-10 steps
# task 1
python3 main.py --steps 10 --world_size 4 --num_workers 2 --port 29502 --output_path ./output/cifar100_3 --now_step 1 --lr 0.001 --eval_step 256 --epoch 128 --warmup_epoch 0 --start_fix 20 --timestamp 20211103110524