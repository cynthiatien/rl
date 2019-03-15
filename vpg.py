"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
"""
import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect


def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
    Building a feedforward neural network. We use neural network to represent our policy and value function(if nn_baseline is present).
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = input_placeholder
        for i in range(n_layers):
            x = tf.layers.dense(x, size, activation=activation)

        output_placeholder = tf.layers.dense(x, output_size, activation=output_activation)
    return output_placeholder


def pathlength(path):
    return len(path["reward"])


def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    # print(args)
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


class Agent(object):
    def __init__(self, env_name, computation_graph_args, sample_trajectory_args, estimate_return_args=None, model_save_args=None):
        super(Agent, self).__init__()
        self.env_name = env_name
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        if estimate_return_args:
            self.gamma = estimate_return_args['gamma']
            self.reward_to_go = estimate_return_args['reward_to_go']
            self.nn_baseline = estimate_return_args['nn_baseline']
            self.normalize_advantages = estimate_return_args['normalize_advantages']
        self.eps = 1e-8
        if model_save_args:
            self.model_dir = model_save_args['model_dir']
            # print("Model dir: ", self.model_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def init_tf_sess(self, savr=True):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()  # equivalent to `with self.sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101
        self.sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.startswith("local")]))
        if savr:
            self.saver = tf.train.Saver()

    def define_placeholders(self):
        """
        Defining the placeholders for (batch) observations, actions and advantage values.
        """
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n

    def policy_forward_pass(self, sy_ob_no):
        """
        Feedforwarding observations throughout our neural network. For discrete action space, we return logits(raw output of neural network), for continuous action space, we return mean and log_std.
        """
        if self.discrete:
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, "discrete_policy",
                                     self.n_layers, self.size, activation=tf.nn.relu)
            return sy_logits_na
        else:
            sy_mean = build_mlp(sy_ob_no, self.ac_dim, "continuous_policy_mean",
                                self.n_layers, self.size, activation=tf.nn.relu)

            sy_logstd = tf.get_variable("continuous_policy_std", shape=[self.ac_dim])
            return (sy_mean, sy_logstd)

    def sample_action(self, policy_parameters):
        """
        Sampling an action from policy distribution. For discrete action space, we sample from the categorical distribution. For continuous action space, we sample from a normal distribution and construct the action with mean and log_std(taking an exp) parameters.
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, num_samples=1), axis=1)
        else:
            sy_mean, sy_logstd = policy_parameters
            z = tf.random_normal(shape=tf.shape(sy_mean))
            sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * z
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """
        Computing the log probability of chosen actions by the policy.
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_ac_na = tf.one_hot(sy_ac_na, self.ac_dim)
            sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sy_ac_na, logits=sy_logits_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_z = (sy_mean - sy_ac_na) / tf.exp(sy_logstd)
            sy_logprob_n = 0.5 * tf.reduce_mean(sy_z ** 2, axis=1)

        return sy_logprob_n

    def build_computation_graph(self):
        """
        Building computation graph for policy gradient algorithm.
        """
        # Defining placeholders for obs/states, actions and advantage values.
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()
        # Computing the logits.
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # Sampling an action according to our policy.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # Computing log_probs of chosen actions.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        # Defining the loss function.
        # http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
        loss = tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        if self.nn_baseline:
            # Create the value network.
            self.baseline_prediction = tf.squeeze(build_mlp(
                self.sy_ob_no,
                1,
                "nn_baseline",
                n_layers=self.n_layers,
                size=self.size))
            # Placeholder for target values which will be used in the loss function for value network.
            self.sy_target_n = tf.placeholder(dtype=tf.float32,
                                              shape=[None],
                                              name='sy_target_n')
            # Define the loss function for value network. Basically MSE loss.
            baseline_loss = tf.reduce_mean((self.baseline_prediction - self.sy_target_n) ** 2)
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)

    def sample_trajectories(self, itr, env):
        """
        Collect paths until we have enough timesteps.
        """
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None]})
            # print("Action: ", ac)
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32)}
        return path

    def sum_of_rewards(self, re_n):
        """
        Monte Carlo estimation of Q values.
        """
        rewards = []
        if self.reward_to_go:
            for re_path in re_n:
                # Per path calculate the estimated rewards for the trajectory
                path_est = []

                # Per time step in the path calculate the reward to go
                for i, re in enumerate(re_path):
                    # Find the len of rtg.
                    reward_to_go_len = len(re_path) - i
                    # Calculate the discount rates.
                    g = np.power(self.gamma, np.arange(reward_to_go_len))
                    # Multiply discount rates with actual rewards and sum.
                    re_to_go = np.sum(g * re_path[i:])
                    path_est.append(re_to_go)

                # Append the path's array of estimated returns
                rewards.append(np.array(path_est))
        else:
            for reward_path in re_n:
                t_prev = np.arange(len(reward_path))
                # Calculate the discount rates.
                gamma = np.power(self.gamma, t_prev)
                # Calculate the discounted total reward.
                discounted_total_reward = np.sum(reward_path * gamma)
                path_r = discounted_total_reward * np.ones_like(reward_path)
                rewards.append(path_r)

        q_val = np.concatenate(rewards)
        return q_val

    def compute_advantage(self, ob_no, q_n):
        """
        Computes advantages by (possibly) subtracting a baseline from the estimated Q values. If not nn_baseline, we just return q_n.
        """

        if self.nn_baseline:
            b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
            # Match the statistics.
            b_n = np.mean(q_n) + np.std(q_n) * b_n
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
        Estimating the returns over a set of trajectories.
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + self.eps)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n, epoch):
        """
        Updating parameters of policy and value function(if nn_baseline).
        """
        if self.nn_baseline:
            # Computing targets for value function.
            target_n = (q_n - np.mean(q_n)) / (np.std(q_n) + self.eps)
            # Updating the value function.
            self.sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no: ob_no,
                                                              self.sy_target_n: target_n})
        # Updating the policy function.
        self.sess.run([self.update_op], feed_dict={self.sy_ob_no: ob_no,
                                                   self.sy_ac_na: ac_na,
                                                   self.sy_adv_n: adv_n})

        # Save the model after updating. No check for the improvement :)
        self.saver.save(self.sess, os.path.join(self.model_dir, "model"), global_step=epoch)


def train_PG(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        reward_to_go,
        animate,
        logdir,
        model_dir,
        normalize_advantages,
        nn_baseline,
        n_layers,
        size):

    start = time.time()

    setup_logger(logdir, locals())

    env = gym.make(env_name)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
    }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }
    model_save_args = {
        'model_dir': model_dir

    }

    agent = Agent(env_name, computation_graph_args, sample_trajectory_args, estimate_return_args, model_save_args=model_save_args)

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        agent.update_parameters(ob_no, ac_na, q_n, adv_n, itr)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = os.path.join('models', logdir)
    logdir = os.path.join('data', logdir)

    if not (os.path.exists(model_dir)):
        os.makedirs(model_dir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    train_PG(
        exp_name=args.exp_name,
        env_name=args.env_name,
        n_iter=args.n_iter,
        gamma=args.discount,
        min_timesteps_per_batch=args.batch_size,
        max_path_length=max_path_length,
        learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go,
        animate=args.render,
        logdir=os.path.join(logdir),
        model_dir=model_dir,
        normalize_advantages=not(args.dont_normalize_advantages),
        nn_baseline=args.nn_baseline,
        n_layers=args.n_layers,
        size=args.size
    )


if __name__ == "__main__":
    main()
