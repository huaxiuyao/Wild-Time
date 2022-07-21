#CORAL
python main.py --dataset=yearbook --method=coral --coral_lambda=0.9 --momentum=0.99 --num_groups=4 --group_size=4 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --random_seed=1 --split_time=1970 --log_dir=./checkpoints

#GroupDRO
python main.py --dataset=yearbook --method=groupdro --num_groups=4 --group_size=4 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --random_seed=1 --split_time=1970 --log_dir=./checkpoints

#IRM
python main.py --dataset=yearbook --method=irm --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=4 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --random_seed=1 --split_time=1970 --log_dir=./checkpoints

#ERM
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --random_seed=1 --split_time=1970 --log_dir=./checkpoints

#LISA
python main.py --dataset=yearbook --method=erm --lisa --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=100 --lr=0.001  --eval_next_timesteps=10 --random_seed=1 --split_time=1970 --log_dir=./checkpoints

#A-GEM
python main.py --dataset=yearbook --method=agem --buffer_size=1000 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --split_time=1970 --random_seed=1 --log_dir=./checkpoints

#EWC
python main.py --dataset=yearbook --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --split_time=1970 --random_seed=1 --log_dir=./checkpoints

#Fine-tuning
python main.py --dataset=yearbook --method=ft --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --split_time=1970 --random_seed=1 --log_dir=./checkpoints

#SI
python main.py --dataset=yearbook --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --eval_next_timesteps=10 --split_time=1970 --random_seed=1 --log_dir=./checkpoints