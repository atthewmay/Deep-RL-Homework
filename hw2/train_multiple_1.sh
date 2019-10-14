#!/bin/bash

#First training the normal single ball w/o any death penalty, using both relative position and not using it
python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000 -lr 5e-3 --output_activation tf.tanh --exp_name GB_sb_ndp --save_models --save_best_model --script_optimizing_dir GB_multi_train
# Quickly runs into the corner, and sometimes gets stuck if the ball is coming from the other corner. It does similarly to using the death penalty from before.

python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000 -lr 5e-3 --output_activation tf.tanh --exp_name GB_sb_ndp_rp -rp --save_models --save_best_model --script_optimizing_dir GB_multi_train
# This one runs away even faster into the corner, more than the original, or maybe the same speed. but it gets stuck in one of the corners! Must have learned too fast or something.
# These two above seem quite the same actually...


# Now running the reward_circle with relative position to compare against what we already have. On train_multiple_2 we have a smaller learning rate
python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000 -lr 5e-3 --output_activation tf.tanh --exp_name GB_sb_ndp_rc_rp -rc -rp --save_models --save_best_model --script_optimizing_dir GB_multi_train
# compared to the one that does not supply relative position, this one runs away from the enemy more effectively, but spends less time in the middle.
# I wonder if it somehow made it easier to learn to run away when it was given the relative positions.



# To run these, do

# python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000  --output_activation tf.tanh --exp_name GB_sb_ndp_rc_rp --run_model_only my_save_loc/GB_multi_train/GB_sb_ndp_GB_game_29-09-2019_00-43-06.ckpt --render
# python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000  --output_activation tf.tanh --exp_name GB_sb_ndp_rc_rp -rc -rp --run_model_only my_save_loc/GB_multi_train/GB_sb_ndp_rp_GB_game_29-09-2019_02-22-36.ckpt --render
# python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000  --output_activation tf.tanh --exp_name GB_sb_ndp_rc_rp -rc -rp --run_model_only my_save_loc/GB_multi_train/GB_sb_ndp_rc_rp_GB_game_29-09-2019_04-00-29.ckpt --render


