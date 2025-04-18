from sklearn.kernel_approximation import RBFSampler
import numpy as np
import argparse

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    logits = logits - np.max(logits,axis=axis, keepdims=True)

    logits = np.exp(logits)

    return logits/np.sum(logits, axis=axis, keepdims=True)


def compute_action_distribution(theta, phis):
    """ compute probability distribution over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: probability distribution over actions that is computed via softmax (shape 1 x |A|)
    """
    
    logits = theta.T @ phis # (1 x d TIMES d x |A| = 1 x |A|)

    return compute_softmax(logits, axis=1) # run softmax over the actions


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """
    #print("theta shape", theta.shape)
    #print("phis shape", phis.shape)
    #print("action idx", action_idx)
    action_distribution = compute_action_distribution(theta, phis) # (1 x |A|)
    #print("distribution shape", action_distribution.shape)
    chosen_phi = (phis[:,action_idx]).T.reshape((-1, 1)) # (d x 1)
    #print("chosen_phi shape", chosen_phi.shape)
    result_of_expectation_over_phis = (action_distribution @ phis.T).T # (d x 1)
    #print("result shape", result_of_expectation_over_phis.shape)
    return chosen_phi - result_of_expectation_over_phis # (d x 1)


def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)

    Note: don't forget to take into account that trajectories might have different lengths
    """

    # TODO
    d = grads[0][0].shape[0]
    N = len(grads)
    outer_sum_result = np.zeros((d,d))
    for trajectory in grads:
        H = len(trajectory)
        inner_sum_result = np.zeros((d,d))
        for gradient in trajectory:
            inner_sum_result = inner_sum_result + (gradient @ gradient.T)
        inner_sum_result = inner_sum_result / H
        outer_sum_result = outer_sum_result + inner_sum_result
    outer_sum_result = outer_sum_result / N
    #print("oingus", outer_sum_result)
    return outer_sum_result + (lamb * np.identity(d))


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: ist of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """
    N = len(grads)
    # baseline calculation
    baseline = 0
    for traj in rewards: # we assume rewards has shape (N, H, 1), where H is
                         # dynamic, being the length of each trajectory
        baseline += np.sum(np.array(traj))
    baseline = baseline / N
    trajectory_idx = 0
    outest_sum_result = 0
    while trajectory_idx < N:
        trajectory = grads[trajectory_idx]
        H = len(trajectory)
        gradient_idx = 0 
        first_nested_sum_result = 0
        while gradient_idx < H:
            gradient = trajectory[gradient_idx]
            reward_timestep = gradient_idx
            reward_trajectory = np.array(rewards[trajectory_idx])
            innest_sum_result = np.sum(reward_trajectory[reward_timestep:]) - baseline
            first_nested_sum_result += (gradient * innest_sum_result)
            gradient_idx += 1
        first_nested_sum_result = first_nested_sum_result / H
        outest_sum_result += first_nested_sum_result
        trajectory_idx += 1
    outest_sum_result = outest_sum_result / N
    
    return outest_sum_result.reshape((-1,1))




def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    return np.sqrt( delta / ((((v_grad.T)@np.linalg.inv(fisher))@v_grad) + 1e-6))



def get_args():
    parser = argparse.ArgumentParser(description='Imitation learning')

    # general + env args
    parser.add_argument('--data_dir', default='./data', help='dataset directory')
    parser.add_argument('--env', default='CartPole-v0', help='environment')
    
    # learning args
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_dataset_samples', type=int, default=10000, help='number of samples to start dataset off with')
    
    # DAGGER args
    parser.add_argument('--dagger', action='store_true', help='flag to run DAGGER')
    parser.add_argument('--expert_save_path', default='./learned_policies/NPG/expert_theta.npy')
    parser.add_argument('--num_rollout_steps', type=int, help='number of steps to roll out with the policy')
    parser.add_argument('--dagger_epochs', type=int, help='number of steps to run dagger')
    parser.add_argument('--dagger_supervision_steps', type=int, help='number of epochs for supervised learning step within dagger')
    
    # model saving args
    parser.add_argument('--policy_save_dir', default='./learned_policies', help='policy saving directory')
    parser.add_argument('--state_to_remove', default=None, type=int, help='index of the state to remove')
    
    return parser.parse_args()