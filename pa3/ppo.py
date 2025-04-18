from argparse import ArgumentParser
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import ActorCritic, ContinuousActorCritic
from tqdm import tqdm
import pathlib


def compute_gae_returns(rewards, values, dones, gamma, gae_lambda):
    """
    Returns the advantages computed via GAE and the discounted returns. 

    Instead of using the Monte Carlo estimates for the returns,
    use the computed advantages and the value function
    to compute an estimate for the returns. 
    
    Hint: How can you easily do this if lambda = 1?

    :param rewards: The reward at each state-action pair
    :param values: The value estimate at the state
    :param dones: Whether the state is terminal/truncated
    :param gamma: Discount factor
    :param gae_lambda: lambda coef for GAE
    """       
    rewards_shape = rewards.shape
    t = len(rewards) - 1
    advantage_t = 0.0
    gae_advantages = torch.zeros(rewards_shape, dtype=torch.float32)
    #print("r", rewards.shape)
    #print("val", values.shape)
    #print("dones", dones.shape)
    #print("gamma", gamma)
    #print("gae_lambda", gae_lambda)

    while t > -1:    
        ###
        # implementation code goes here
        delta_t = rewards[t] + (gamma * values[t+1] * (1 - dones[t])) - values[t]
        #print("detla t", delta_t.shape)
        advantage_t = delta_t + (gamma * gae_lambda * (1- dones[t]) * advantage_t) 
        #print("shape adv", advantage_t.shape)
        #print("advantage gae", gae_advantages.shape)
        gae_advantages[t] = advantage_t
        # end of code implementation block
        ###
        t -= 1
    discounted_returns = gae_advantages + values[:len(values)-1]
    return gae_advantages, discounted_returns


def ppo_loss(agent: ActorCritic, states, actions, advantages, logprobs, returns, clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5) -> torch.Tensor:
    """
    Compute the PPO loss. You can combine the policy, value and entropy losses into a single value. 

    :param policy: The policy network
    :param states: States batch
    :param actions: Actions batch
    :param advantages: Advantages batch
    :param logprobs: Log probability of actions
    :param returns: Returns at each state-action pair
    :param clip_ratio: Clipping term for PG loss
    :param ent_coef: Entropy coef for entropy loss
    :param vf_coef: Value coef for value loss
    """  
    returned_actions, returned_action_log_probabilities, returned_entropies, returned_state_values = agent.action_value(states,actions)
    probability_ratio = torch.exp(returned_action_log_probabilities - logprobs)

    unclipped_term = probability_ratio * advantages
    clipped_term = torch.clip(probability_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -1 * torch.mean(torch.min(unclipped_term, clipped_term))

    value_loss = torch.mean((returned_state_values - returns)**2)

    entropy_loss = torch.mean(returned_entropies)
    return policy_loss + (vf_coef * value_loss) - (ent_coef * entropy_loss)

def make_env(env_id, **kwargs):
    def env_fn():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        return env
    return env_fn


def train(
    env_id="CartPole-v0",
    epochs=500,
    num_envs=4,
    gamma=0.99,
    gae_lambda=0.9,
    lr=3e-4,
    num_steps=128,
    minibatch_size=32,
    clip_ratio=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    update_epochs=3,
    seed=42,
    checkpoint=False,
    max_grad_norm=0.5,
):
    """
    Returns trained policy. 
    """

    # Try not to modify this
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    eval_env = make_env(env_id)()
    
    if isinstance(env.single_action_space, gym.spaces.Discrete):
        policy = ActorCritic(env.single_observation_space.shape[0], env.single_action_space.n).to(device)
    else:
        policy = ContinuousActorCritic(env.single_observation_space.shape[0], env.single_action_space.shape[0])
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    states = torch.zeros((num_steps, num_envs) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + env.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps + 1, num_envs)).to(device)

    obs, _ = env.reset(seed = np.random.randint(2**30))
    obs = torch.from_numpy(obs).float().to(device)

    pathlib.Path(f"learned_policies/{env_id}/").mkdir(parents=True, exist_ok=True)
    
    for iteration in tqdm(range(1, epochs + 1)):

        # TODO: Collect num_steps transitions from env and fill in the tensors for states, actions, ....
        for step_idx in range(num_steps):
            action, log_prob, entropy, state_value = policy.action_value(obs)
            #print("what",action.shape)
            
            
            next_obs, reward, done, trunc, info = env.step(np.array(action))

            #action.requires_grad_()
            #obs.requires_grad_()

            

            states[step_idx] = obs
            actions[step_idx] = action
            logprobs[step_idx] = log_prob
            rewards[step_idx] = torch.tensor(reward).float().to(device)
            dones[step_idx]= torch.tensor(done).float().to(device)
            values[step_idx] = state_value

            obs = torch.tensor(next_obs).float().to(device)
        
        with torch.no_grad():
            values[-1] = policy.value(obs)

            # TODO: Compute Advantages and Returns
            advantages, returns = compute_gae_returns(rewards,values,dones,gamma,gae_lambda)

        # TODO: Perform num_steps / minibatch_size gradient updates per update_epoch
        # Perform gradient updates using minibatches and PPO loss

        
        """
        for _ in range(update_epochs):
            # Shuffle transitions to create mini-batches
            indices = torch.randperm(num_steps * num_envs)
            for i in range(0, len(indices), minibatch_size):
                batch_indices = indices[i:i + minibatch_size]
                batch_states = states.view(-1, *states.shape[2:])[batch_indices]
                batch_actions = actions.view(-1, *actions.shape[2:])[batch_indices]
                batch_logprobs = logprobs.view(-1)[batch_indices]
                batch_advantages = advantages.view(-1)[batch_indices]
                batch_returns = returns.view(-1)[batch_indices]

                # Calculate the loss and backpropagate
                loss = ppo_loss(
                    agent=policy,
                    states=batch_states,
                    actions=batch_actions,
                    advantages=batch_advantages,
                    logprobs=batch_logprobs,
                    returns=batch_returns,
                    clip_ratio=clip_ratio,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef
                )
                print("ungus")
                optimizer.zero_grad()
                loss.backward()
                print("wungus")
                
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()"""
        batch_sta = states.reshape((-1,) + env.single_observation_space.shape)
        batch_act = actions.reshape((-1,) + env.single_action_space.shape)
        batch_log = logprobs.reshape((-1))
        batch_adv = advantages.reshape((-1))
        batch_ret = returns.reshape((-1))
        for _ in range(update_epochs):
            # Shuffle transitions to create mini-batches
            indices = torch.randperm(num_steps * num_envs)
            
            for i in range(0, num_steps * num_envs, minibatch_size):
                batch_indices = indices[i:i + minibatch_size]
                batch_states = batch_sta[batch_indices].detach().clone()
                batch_actions = batch_act[batch_indices].detach().clone()
                batch_logprobs = batch_log[batch_indices].detach().clone()
                batch_advantages = batch_adv[batch_indices].detach().clone()
                batch_returns = batch_ret[batch_indices].detach().clone()

                #print("Batch States Requires Grad: ", batch_states.requires_grad)
                #print("Batch Actions Requires Grad: ", batch_actions.requires_grad)
                #print("Batch Logprobs Requires Grad: ", batch_logprobs.requires_grad)
                #print("Batch Advantages Requires Grad: ", batch_advantages.requires_grad)
                #print("Batch Returns Requires Grad: ", batch_returns.requires_grad)

                # Calculate the loss and backpropagate
                loss = ppo_loss(
                    agent=policy,
                    states=batch_states,
                    actions=batch_actions,
                    advantages=batch_advantages,
                    logprobs=batch_logprobs,
                    returns=batch_returns,
                    clip_ratio=clip_ratio,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef
                )

                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Perform a backward pass for this batch
                
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()  # Update the model parameters


        # Clip the gradient
        #nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        # Uncomment for eval/checkpoint
        if iteration % 10 == 0: 
            print(f"Eval Reward {iteration}:", (val(policy, eval_env)))
            if checkpoint:
                torch.save(policy, f"learned_policies/{env_id}/model_{iteration}.pt")
        
    
    return policy


def val(model, env, num_ep=100):
    rew = 0
    for i in range(num_ep):
        done = False
        obs, _ = env.reset(seed=np.random.randint(2**30))
        obs = torch.from_numpy(obs).float()
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = model.action_value(obs)
            obs, reward, done, trunc, _ = env.step(action.cpu().numpy())
            obs = torch.from_numpy(obs).float()
            done |= trunc
            rew += reward

    return rew / num_ep

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=3, help="Number of epochs over data every iteration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    # Feel free to add or remove parameters

    args = parser.parse_args()  
    policy = (train(**vars(args)))
    torch.save(policy, f"learned_policies/{args.env_id}/model.pt")