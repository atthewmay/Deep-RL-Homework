from layers_model import layers_model
import tensorflow as tf 

def train_model():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data_file', type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--regularization_weight", type=float, default=0.001)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument('--resume', action='store_true')


    parser.add_argument('--dagger', action='store_true')
    # parser.add_argument("--expert_policy_file", type=str, help="required if running dagger")
    parser.add_argument("--dagger_iter", type=int, default=30)
    parser.add_argument("--rollouts_per_dagger", type=int, default=50)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--save_performance_data", action = 'store_true')

    args = parser.parse_args()
    args.envname = "".join(args.training_data_file.split(".")[:1])

    args.expert_policy_file = 'experts/'+args.training_data_file # SUCH Jank solution


    tf.reset_default_graph()

    mymodel = layers_model('expert_data/'+args.training_data_file,reg_scale=args.regularization_weight)

    if args.dagger:
        mymodel.dagger_train_model(args.dagger_iter, args.rollouts_per_dagger, args.envname, args.expert_policy_file, save_data_agg = None, render_runs = False, iterations = 1000,
         save_string = args.envname+'_model_'+str(args.batches)+'-batches' + '_dagger_' + str(args.dagger_iter), 
         resume = args.resume, lr = args.lr, batch_size = args.batch_size, save_performance_data = args.save_performance_data)
    else:
        mymodel.train_model(args.batches,save_string = args.envname+'_model_'+str(args.batches)+'-batches',resume = args.resume, lr = args.lr,
            batch_size = args.batch_size, save_single_performance_data = args.save_performance_data, envname = args.envname)

if __name__ == '__main__':
    train_model()