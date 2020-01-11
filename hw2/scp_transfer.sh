
#!/bin/bash


suffixes='data-00000-of-00001 index meta'
for suffix in $suffixes
# first we are going to transfer the models and then the data	
	do 
		gcloud compute scp --recurse instance-1:/home/matthewhunt/deep_rl_course/homework/hw2/my_save_loc/gb_discrete/$1.ckpt.$suffix Coding/RL_Stuff/berkley_rl_course/homework/hw2/my_save_loc/$2
	done 
# Next we'll transfer the data

gcloud compute scp --recurse instance-1:/home/matthewhunt/deep_rl_course/homework/hw2/data/gb_discrete/$1 Coding/RL_Stuff/berkley_rl_course/homework/hw2/data/$2

