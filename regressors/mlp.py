#=====================
# multilayer neural network
# Author: Liujiandu
# Date: 2018/1/10
#=====================
import tensorflow as tf

def fc(x, shape, scope, actf=tf.nn.tanh):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [shape[1]], initializer=tf.constant_initializer(0.01))
        z = actf(tf.matmul(x,w)+b)
    return z


#network
class MLP(object):
    def __init__(self, x_dim, y_dim, hidden_dim, scope='mlp', reuse=False):
        """
        :Param hidden_dim:
            hidden layer neuron number, list like
        """
        self.hidden_dim=hidden_dim
        self.x = tf.placeholder(tf.float32, [None, x_dim])
        self.y_ = tf.placeholder(tf.float32,[None, y_dim])
        
        with tf.variable_scope(scope, reuse=reuse):
            f = fc(self.x, [x_dim, self.hidden_dim[0]], 'fc1',actf=tf.nn.tanh)
            for i in range(len(self.hidden_dim)-1):
                f = fc(f, [self.hidden_dim[i], self.hidden_dim[i+1]], 'fc'+str(i),actf=tf.nn.tanh)
            self.y = fc(f, [self.hidden_dim[-1], y_dim], 'y', actf=tf.identity)

            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y-self.y_),reduction_indices = [1]))

            #train
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        #initialize
        #self.init = tf.global_variables_initializer()
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def update(self, x, y_): 
        self.sess.run(self.train_step, feed_dict=({self.x:x, self.y_:y_}))

    def get_loss(self, x, y_):    
        return self.sess.run(self.loss, feed_dict={self.x:x, self.y_: y_})
     
    def fit(self, x, y):
        """
        train mlp to fit sampled data points(x, y)

        Parameters:
        ------------
        x:2d-array
            input of sampled data
        y:2d-array
            output of sampled data
        """
        assert x.ndim==2, 'x must be 2D array'
        assert y.ndim==2, 'y must be 2D array'

        self.sess.run(self.init)
        for i in range(10000):
            self.update(x, y)
            if i%1000==0: 
                loss=self.get_loss(x,y.reshape((-1,1)))
                print loss
                if loss<1e-4:
                    break

    def predict(self, x):    
        """
        predict by trained mlp
        
        Parameters:
        x: 2d-array

        """
        mu = self.sess.run(self.y, feed_dict={self.x:x})
        sigma = None
        return mu, sigma

    def set_params(self, **params):
        pass
