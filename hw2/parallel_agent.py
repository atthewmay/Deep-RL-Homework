import time
import psutil
import ray
import sys
import tensorflow as tf

num_cpus = psutil.cpu_count(logical=False)
print('the number of cpus is ' + str(num_cpus))
ray.init(num_cpus=num_cpus)

filename = '/tmp/model'

def test_fn():
    x = 2
    return x

@ray.remote
class Parallel_Actor(object):
    def __init__(self,i):
        """This will work a litle differently. We will establish the number of workers up front.
        Then we will first build a model graph and then load-up/update our model parameters we'll be using on each
        actor. Then when it comes time to run the actors, we will invoke all of them using somehting like:
            ray.get([actor.sample_trajectories().remote() for actor in actors] (actors is defined in the
            train_pg function). """
        # Pin the actor to a specific core if we are on Linux to prevent
        # contention between the different actors since TensorFlow uses
        # multiple threads.
        if sys.platform == 'linux':
            psutil.Process().cpu_affinity([i])
        # Load the model and some data.
#        self.model = tf.keras.models.load_model(filename)
#        mnist = tf.keras.datasets.mnist.load_data()
#        self.x_test = mnist[1][0] / 255.0
        self.i = i

    def sample_trajectories(self, env):
        """This will perform trajectory sampling. It will use a loaded model."""
#        return self.model.predict(self.x_test)
        return self.i*2

    def test_test(self):
        return test_fn()


actors = [Parallel_Actor.remote(i) for i in range(num_cpus)]

# Time the code below.
time.sleep(1)
start = time.time()
# Parallelize the evaluation of some test data.
results = [ray.get(actor.test_test.remote()) for actor in actors]
#results = ray.get([ray.get(actor.test_test.remote()) for actor in actors])
end = time.time()
duration = end-start

print('so the full results thing is')
print(results)

print('the full duration is '+str(duration))
