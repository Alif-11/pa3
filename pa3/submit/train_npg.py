import gymnasium as gym
import numpy as np
import npg_utils
import matplotlib.pyplot as plt
import os

def sample(theta, env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []
    for n in range(N):
        trajectory_grads = []
        trajectory_rewards = []
        
        # TODO Get initial state
        observation = env.reset(seed=np.random.randint(2**31))[0]
        for t in range(200):
            # TODO Extract features, get trajectory_grads and get trajectory_rewards
            phis = npg_utils.extract_features(observation,num_actions=2) # only 2 actions for CartPole
            action_distribution = npg_utils.compute_action_distribution(theta, phis)
            chosen_action = np.random.choice(2,p=action_distribution.reshape((-1)))
            current_gradient = npg_utils.compute_log_softmax_grad(theta, phis,chosen_action)
            next_observation, current_reward, done, truncated, info = env.step(chosen_action)

            trajectory_grads.append(current_gradient)
            trajectory_rewards.append(current_reward)

            if done:
                break

            observation = next_observation # update observation to become the 
                                           # next observation

        total_rewards.append(trajectory_rewards)
        total_grads.append(trajectory_grads)


    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100,1)
    env = gym.make('CartPole-v0')
    

    episode_rewards = []

    for t in range(T):
        print(f"Iteration {t}")
        gradients, rewards = sample(theta, env, N)
        fisher = npg_utils.compute_fisher_matrix(gradients,lamb)
        v_grad = npg_utils.compute_value_gradient(gradients,rewards)
        eta = npg_utils.compute_eta(delta, fisher, v_grad)

        theta = theta + (eta * (np.linalg.inv(fisher) @ v_grad))

        avg_reward = []
        for reward_list in rewards:
            avg_reward.append(np.sum(np.array(reward_list)))
        avg_reward = np.mean(avg_reward)
        episode_rewards.append(avg_reward)
        # TODO Update theta according to handout, and record rewards

    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    theta_dir = 'learned_policies/NPG'
    os.makedirs(theta_dir, exist_ok=True)
    np.save(os.path.join(theta_dir, 'expert_theta.npy'), theta)

    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plot_dir = './plots'
    plt.savefig(os.path.join(plot_dir, "rewards"))
    plt.show()
