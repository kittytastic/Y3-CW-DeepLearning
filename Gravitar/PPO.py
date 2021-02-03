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
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
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
       

    def chooseAction(self, state):
        mu = self.actor(state)
        dist = torch.distributions.Categorical(mu)
        action = dist.sample()
        return action, dist

    def getCriticFor(self, state):
        return self.critic(state)

    def learn(self, stuff):
        pass


def testReward(env, actor_critic):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action, _ = actor_critic.chooseAction(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

def accrueExperience(env, actor_critic, steps=128):

    rewards = []
    states = []
    actions = []
    probs = []
    masks = []
    values = []

    state = env.reset()
    for e in range(steps):
        state = torch.FloatTensor(state)
        action, action_distribution = actor_critic.chooseAction(state)
        estimated_value = actor_critic.getCriticFor(state)

        next_state, reward, done, _ = env.step(action.cpu().numpy())
        
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

    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'probs':probs, 'values':values}, next_value

def proccessExperiences(next_value, raw_experience, gamma=0.99, tau=0.95):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)
    values += [next_value]

    # Calculate General advantage estimation and returns for each step
    gae = 0
    retruns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] - values[step] + gamma*values[step+1]*masks[step] 
        gae = delta + gamma * tau * masks[step] * gae
        retruns.insert(0, gae + values[step])
    
    return {**raw_experience, 'returns':retruns}

def miniBatchIter(mini_batch_size, mini_batch_count, experiences):
    batch_size = len(experiences[experiences.keys()[0]])
    for _ in range(mini_batch_count):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield {key: field[rand_ids, :] for key, field in experiences}


def teachActorCritic(actor_critic, experiences, epochs=10):

    for e in range(epocs):
        al, cl = actor_critic.learn(experiences)
    


#env_name = 'Gravitar-ram-v0'
env_name = 'CartPole-v0'
env = gym.make(env_name)


assert(len(env.observation_space.shape)==1)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n
print("inputs: %d   outputs: %d   for: %s"%(num_inputs, num_outputs, env_name))

model = ActorCritic(num_inputs, num_outputs, 128)

raw_experience, next_value = accrueExperience(env, model, 2)

experience = proccessExperiences(next_value, raw_experience)
#print(experience)
#print(raw_experience)