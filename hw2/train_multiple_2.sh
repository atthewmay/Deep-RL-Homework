#!/bin/bash

# Trying a 2 ball experimetn w/ reward circle and relative position. 
python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000 -lr 5e-3 --output_activation tf.tanh --exp_name GB_2b_ndp_rc_rp -rc -rp --num_enemies 2 --save_models --save_best_model --script_optimizing_dir GB_multi_train
# This one is actually pretty cool. Its a tight board, and it huddles int he coner too much I think. but survives quite well!
# Now running the reward_circle with relative position to compare against what we already have. On train_multiple_2 we have a smaller learning rate

python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000 -lr 5e-4 --output_activation tf.tanh --exp_name GB_sb_ndp_rc_rp -rc -rp --save_models --save_best_model --script_optimizing_dir GB_multi_train

#To run these# So this one has the slower learning rate. I think it's got the right idea (stays in middle too much), and just needed to train more!


# python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000 -lr 5e-3 --output_activation tf.tanh --exp_name GB_2b_ndp_rc_rp -rc -rp --num_enemies 2 --run_model_only my_save_loc/GB_multi_train/GB_2b_ndp_rc_rp_GB_game_29-09-2019_00-44-08.ckpt --render
# python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 200 -l 2 -s 32 -b 60000  --output_activation tf.tanh --exp_name GB_sb_ndp_rc_rp -rc -rp --run_model_only my_save_loc/GB_multi_train/GB_sb_ndp_rc_rp_GB_game_29-09-2019_02-32-14.ckpt --render