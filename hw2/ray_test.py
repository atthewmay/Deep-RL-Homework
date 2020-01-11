import psutil
import ray
import sys
import tensorflow as tf

num_cpus = psutil.cpu_count(logical=False)
print('the number of cpus is ' + str(num_cpus))
ray.init(num_cpus=num_cpus)

filename = '/tmp/model'

@ray.remote
class Model(object):
    def __init__(self, i):
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
    def evaluate_next_batch(self):
        # Note that we reuse the same data over and over, but in a
        # real application, the data would be different each time.
#        return self.model.predict(self.x_test)
        return self.i*2

actors = [Model.remote(i) for i in range(num_cpus)]

# Time the code below.

# Parallelize the evaluation of some test data.
results = ray.get([actor.evaluate_next_batch.remote() for actor in actors])

print('so the full results thing is')
print(results)
