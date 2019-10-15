# CS294-112 HW 2: Policy Gradient

In this homework, we implement the policy gradient algorithm for both discrete and continuous action spaces and test it on several environments. We also implement various methods of reducing the variance of the gradient of the policy, such as normalization of training data, reward-to-go, and neural-network baselines. 

The code for setting up the policy-gradient-based agent and training that agent is found in train_pg_f18.py. This code was provided by the instructors with portions to fill in. It's a work in progress, and I've left many notes-to-self in the comments.

#### Overview of Policy Gradient Algorithm
In policy gradient, we use a neural network to represent the policy of an actor. That is, the network parameterizes the probability distribution of outputs (potential actions the actor could take) given the inputs (the state or observation the actor sees at each timestep). We directly optimize the network that forms the agent's policy by taking the gradient of the expectation of the reward the agent garners using its current policy ( E_all_paths[r(path)] ) and performing some form of stochastic gradient descent. That gradient is taken with respect to the parameters of the neural network policy. 


## Cart Pole Experiments
Using the cart-pole environment from OpenAI gym, we test different batch sizes and several different configurations of the variance-reducing strategies. For each experiment, we used 100 iterations of 2 different batch sizes, and we ran each of these 3 times (-e is 3) so as to plot the mean and standard deviation of the rewards.

These experiments were run using the following commands:  
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna  
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna  
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na  
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna  
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna  
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na  
 
| batch_size = 1000 | batch_size = 5000 |  
| ------------------------- | ------------------------- |  
| ![](result_plots/Figure_1.png) | ![](result_plots/large_batch.png) | 
For the legend: "-rtg" = used reward-to-go, "-dna" = don't normalize advantage 

#### Analysis
The network using reward-to-go seems more solidly convergent on the target of 200 reward. 
Not normalizing the advantage actually helps it to converge faster, which makes sense, because the gradient will be larger, but the variance should also be higher, as the rewards we use to scale grad_log_prob(action) are larger. This higher variance displays itself, as the policy without normalized advantage shows reduced convergence to the optimum. This is the case for both large and small batches.

The larger batches do not converge much faster, if at all. Using tf.reduce_mean() for the loss function means that the gradients should be approximately the same magnitude. However, with larger batches, you expect the estimated gradient to be a better estimate of the true gradient. I Suspect this helps it to converge a little bit faster, and it helps the methods with higher variance converge better to the optimum. Because it should decrease the variance for every method, every method performs better at convergence.


For fun: Demonstration of the cartpole task as learning progresses with little training (n = 3 and 50) and smaller batches (size = 500)
run using: python train_pg_f18.py CartPole-v0 -n #\_iterations -b 500 -e 1 -dna -rtg --exp_name test --save_models --render
So both used reward-to-go and did not normalize advantage

| After 3 SGD iterations | After 50 SGD iterations |  
| ------------------------- | ------------------------- |  
| ![](result_plots/cart_pole_v0_n3_b500.gif) | ![](result_plots/cart_pole_v0_n50_b500.gif) |  
Each time the cart-pole simulation with little training twitches it represents a crash.


