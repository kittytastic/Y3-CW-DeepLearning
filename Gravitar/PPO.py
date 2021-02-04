import torch
from torch import nn
import gym
import collections
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# Python 3.9.1 itemgetter op
def itemgetter(*items):
    if len(items) == 1:
        item = items[0]
        def g(obj):
            return obj[item]
    else:
        def g(obj):
            return tuple(obj[item] for item in items)
    return g

# Need an optimiser for actor critic
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0, lr=3e-4):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=0),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
       

    def chooseAction(self, state):
        mu = self.actor(state)
        dist = torch.distributions.Categorical(mu)
        action = dist.sample()
        return action, dist

    def getCriticFor(self, state):
        return self.critic(state)

    def learn(self, epochs, experience, clip_epsilon=None, mini_batch_size=None, entropy_coeff=None, vf_coeff=None): 
        for _ in range(epochs):
            for mini_experience_batch in miniBatchIter(mini_batch_size, experience):
            
                state, action, old_log_probs, retruns, advantage = itemgetter('states', 'actions', 'log_probs', 'returns', 'advantage' )(mini_experience_batch)
                
                _, dist = self.chooseAction(state)
                value = self.getCriticFor(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0-clip_epsilon, 1.0+clip_epsilon)*advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (retruns - value).pow(2).mean()

                loss =  vf_coeff * critic_loss + actor_loss - entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def testReward(env, actor_critic):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = torch.FloatTensor(state)
        action, _ = actor_critic.chooseAction(state)
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        state = next_state
        total_reward += reward

    return total_reward

def accrueExperience(env, actor_critic, steps=128):

    rewards = []
    states = []
    actions = []
    probs = []
    masks = []
    values = []
    log_probs = []

    state = env.reset()
    for e in range(steps):
        state = torch.FloatTensor(state)
        action, action_distribution = actor_critic.chooseAction(state)
        estimated_value = actor_critic.getCriticFor(state)

        next_state, reward, done, _ = env.step(action.cpu().numpy())

        log_prob = action_distribution.log_prob(action)
        log_probs.append(log_prob)       
        masks.append(int(not done)) # Used to mask out
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        
        probs.append(action_distribution)
        values.append(estimated_value)

        state = next_state
        if done:
            env.reset()

    # We need one extra for GAE calculation
    next_state = torch.FloatTensor(next_state)
    next_value = actor_critic.getCriticFor(next_state)

    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'log_probs': log_probs, 'probs':probs, 'values':values}, next_value

def proccessExperiences(next_value, raw_experience, gamma=None, tau=None):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)
    values = values + [next_value]

    # Calculate General advantage estimation and returns for each step
    gae = 0
    retruns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] - values[step] + gamma*values[step+1]*masks[step] 
        gae = delta + gamma * tau * masks[step] * gae
        retruns.insert(0, gae + values[step])
    
    return {**raw_experience, 'returns':retruns}

def miniBatchIter(mini_batch_size, experiences):
    batch_size = len(experiences[list(experiences.keys())[0]])
    for _ in range(batch_size//mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield {key: field[rand_ids] for key, field in experiences.items()}

    


discount_gamma = 0.99
GAE_tau = 0.95
epochs = 10
timesteps = 128 * 8
mini_batch_size = 32
entropy_coeff = 0.01
vf_coeff = 1
epsilon = 0.2

video_every = 1

env_name = 'Gravitar-ram-v0'
#env_name = 'CartPole-v0'
env_name = 'SpaceInvaders-ram-v0'
env = gym.make(env_name)
env_test = gym.make(env_name)
#env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0, force=True)
env_test = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0, force=True)


assert(len(env.observation_space.shape)==1)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n
print("inputs: %d   outputs: %d   for: %s"%(num_inputs, num_outputs, env_name))

model = ActorCritic(num_inputs, num_outputs, 128)

for i in range(1000):
    raw_experience, next_value = accrueExperience(env, model, steps=timesteps)

    experience = proccessExperiences(next_value, raw_experience, gamma=discount_gamma, tau=GAE_tau)

    returns   = torch.cat(experience['returns']).detach()
    log_probs = torch.Tensor(experience['log_probs']).detach()
    values    = torch.Tensor(experience['values']).detach()
    states = torch.Tensor()
    states    = torch.cat(experience['states']).view(len(experience['states']), -1)
    actions   = torch.Tensor(experience['actions'])
    advantage = returns - values

    experience = {'returns':returns, 'log_probs':log_probs, 'values':values, 'states':states, 'actions':actions, 'advantage':advantage}
    model.learn(epochs, experience, 
        mini_batch_size=mini_batch_size,
        entropy_coeff=entropy_coeff,
        vf_coeff=vf_coeff,
        clip_epsilon=epsilon)

    if i % 10 == 0:
        print(testReward(env_test, model))