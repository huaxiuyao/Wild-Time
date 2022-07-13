#CORAL
python main.py --dataset=huffpost --method=coral --offline --coral_lambda=0.9 --num_groups=3 --group_size=2 --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#GroupDRO
python main.py --dataset=huffpost --method=groupdro --offline --num_groups=3 --group_size=2 --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#IRM
python main.py --dataset=huffpost --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=2 --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#ERM
python main.py --dataset=huffpost --method=erm --offline --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#LISA
python main.py --dataset=huffpost --method=erm --lisa --offline --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints

#A-GEM
python main.py --dataset=huffpost --method=agem --buffer_size=1000 --mini_batch_size=16 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2015 --random_seed=1 --log_dir=./checkpoints

#EWC
python main.py --dataset=huffpost --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=16 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2015 --random_seed=1 --log_dir=./checkpoints

#Fine-tuning
python main.py --dataset=huffpost --method=ft --mini_batch_size=16 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2015 --random_seed=1 --log_dir=./checkpoints

#SI
python main.py --dataset=huffpost --method=si --si_c=0.1 --epsilon=1e-4 --mini_batch_size=16 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2015 --random_seed=1 --log_dir=./checkpoints
