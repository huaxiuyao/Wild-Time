#CORAL
python main.py --dataset=arxiv --method=coral --eval_next_timesteps=3 --split_time=2016 --coral_lambda=0.9 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=5 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#GroupDRO
python main.py --dataset=arxiv --method=groupdro --eval_next_timesteps=3 --split_time=2016 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=5 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#IRM
python main.py --dataset=arxiv --method=irm --eval_next_timesteps=3 --split_time=2016 --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=5 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#ERM
python main.py --dataset=arxiv --method=erm --eval_next_timesteps=3 --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=5 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#LISA
python main.py --dataset=arxiv --method=erm --lisa --eval_next_timesteps=3 --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=5 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#A-GEM
python main.py --dataset=arxiv --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=3 --split_time=2016 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#EWC
python main.py --dataset=arxiv --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=3 --split_time=2016 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#Fine-tuning
python main.py --dataset=arxiv --method=ft --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=3 --split_time=2016 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#SI
python main.py --dataset=arxiv --method=si --si_c=0.1 --epsilon=1e-4 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --eval_next_timesteps=3 --split_time=2016 --num_workers=8 --random_seed=1 --log_dir=./checkpoints