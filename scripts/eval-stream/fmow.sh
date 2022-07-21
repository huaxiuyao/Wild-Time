#CORAL
python main.py --dataset=fmow --method=coral --eval_next_timesteps=6 --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=2 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=7 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#GroupDRO
python main.py --dataset=fmow --method=groupdro --eval_next_timesteps=6 --num_groups=3 --group_size=2 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --split_time=7 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#IRM
python main.py --dataset=fmow --method=irm --eval_next_timesteps=6 --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=2 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=7 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#ERM
python main.py --dataset=fmow --method=erm --eval_next_timesteps=6 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=7 --num_workers=8 --random_seed=1  --log_dir=./checkpoints

#LISA
python main.py --dataset=fmow --method=erm --lisa --eval_next_timesteps=6 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=7 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#A-GEM
python main.py --dataset=fmow --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --eval_next_timesteps=6 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#EWC
python main.py --dataset=fmow --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --eval_next_timesteps=6 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#Fine-tuning
python main.py --dataset=fmow --method=ft --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --eval_next_timesteps=6 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#SI
python main.py --dataset=fmow --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --eval_next_timesteps=6 --split_time=7 --random_seed=1 --log_dir=./checkpoints