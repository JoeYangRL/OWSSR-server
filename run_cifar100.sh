## CIFAR100 B45-1 steps
# task 1
#python3 main.py --base_task_cls 45 --steps 1 --world_size 4 --num_workers 0 --port 29502 --output_path ./output/cifar100_3 --now_step 1 --lr 0.1 --eval_step 256 --epoch 128 --warmup_epoch 0 --start_fix 20 --timestamp 20211108113629
#python3 main.py --base_task_cls 45 --steps 1 --world_size 4 --num_workers 0 --port 29500 --output_path ./output/cifar100 --now_step 1 --lr 0.1 --eval_step 256 --epoch 128 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --batch_size 64 --start_fix 20 --timestamp 20211109151438
## CIFAR100 B55-1 steps
# task 1
#python3 main.py --base_task_cls 55 --steps 1 --world_size 2 --num_workers 2 --port 29555 --output_path ./output/cifar100_pil_detach2 --now_step 1 --lr 0.03 --eval_step 512 --epoch 128 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --threshold 0.0 --batch_size 64 --start_fix 10 --timestamp 20211118155930 --use-ema
# task 2
#python3 main.py --base_task_cls 55 --steps 1 --world_size 1 --num_workers 1 --port 29555 --output_path ./output/cifar100_pil_detach2 --now_step 2 --lr 0.03 --eval_step 100 --epoch 1 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --threshold 0.0 --batch_size 64 --start_fix 10 --timestamp 20211118155930 --use-ema

## CIFAR100 B0-10 steps
# task 1
#python3 main.py --steps 10 --world_size 4 --num_workers 0 --port 29502 --output_path ./output/cifar100 --now_step 1 --lr 0.01 --eval_step 256 --epoch 128 --warmup_epoch 0 --start_fix 20 --batch_size 32 --timestamp 20211103110524

## CIFAR100 B0-5 steps
# task 1
#python3 main.py --steps 5 --world_size 2 --num_workers 1 --port 29555 --output_path ./output/cifar100_b0-5steps-fast --now_step 1 --lr 0.03 --eval_step 512 --epoch 64 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --threshold 0.0 --batch_size 64 --start_fix 10 --timestamp 20211124104446

## CIFAR100 B80-1 steps
# task 1
python3 main.py --base_task_cls 80 --steps 1 --world_size 2 --num_workers 1 --port 29554 --output_path ./output/cifar100_80_syncbn --now_step 1 --lr 0.06 --eval_step 512 --epoch 128 --warmup_epoch 0 --lambda_oem 0.1 --lambda_socr 1.0 --threshold 0.0 --batch_size 64 --start_fix 10 --timestamp 20211203154700 --use-ema
