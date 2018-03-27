import tensorflow as tf

class Model():
    def __init__(self):
        self.build_model()

        # hyperparemeters
        self.dense1_input_size = 512
        self.dropout_rate = 0.05
        self.dense2_input_size = 512
        self.dense3_input_size = 512

        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.997

        self.alpha = 1
        self.min_alpha = 0.05
        self.alph_decay = 0.997

        self.gama = 0.99

    def calc_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.min_epsilon, epsilon)

    def calc_alpha(self):
        alpha = self.alpha * self.alpha_decay
        self.alpha = max(self.min_alpha, alpha)

    def build_model(self, state_dim_len, action_dim_len):
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, sate_dim_len])
        self.new_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim_len])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

    def build_q_networks(self):
        with tf.variable_scope('q_network') as scope:
            q_action_values_old = self.generate_network(True)
            q_action_values_new = self.generate_network(False)

    def build_slow_target_network(self):
        with tf.variable_scope('slow_target_network'):
            slow_target_action_values = tf.stop_gradient(
                    self.generate_network(
                        self.new_state,
                        is_trainable=False)
    def get_q_state(self):
        return sess.run(
                self.q_action_values_old,
                feed_dict={ state: old_state[None], is_training: False })

    def init_sess(self):
        self.sess = tf.Session()


    def build_network(self, is_trainable):
        self.dense1 = tf.layers.dense(
                self.state_dim_len,
                self.h1_input_size,
                activation=tf.nn.relu,
                trainable = is_trainable,
                name='dense1')

        self.dropout1 = tf.layers.dropout(
                self.dense1,
                rate=self.dropout_rate,
                trainable=is_trainable,
                name='dropout1')

        self.dense2 = self.layers.dense(
                self.dropout1,
                self.dense2_input_size,
                activation=tf.nn.relu,
                trainable = is_trainable,
                mame='dense2')

        self.dropout2 = tf.layers.droput(
                self.dense2, rate=self.dropout_rate, trainable=is_trainable, name='dropout2')

        self.dense3 = tf.layers.dense(
                self.dropout2,
                self.dense3_input_size,
                action=tf.nn.relu,
                trainable = is_trainable
                name='dense3')

        self.dropout3 = tfl.layers.dense(
                self.dense3,
                rate=self.dropout_rate,
                trainable=is_trainable,
                name='dropout3')

        self.activation_values = tf.squeeze(
                tf.layers.dense(
                    self.dropout3,
                    self.action_dim_len,
                    trainable=is_trainable,
                    name='dense3'))
