import torch
from torch import nn
import gym
import collections
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

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


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0, lr=3e-4):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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


def testReward(env, actor_critic, device):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = torch.FloatTensor(state).to(device)
        action, _ = actor_critic.chooseAction(state)
        next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
        state = next_state
        total_reward += reward

    return total_reward

def accrueExperience(env, actor_critic, state_space_size, device, steps=None):

    rewards = torch.zeros(steps, requires_grad=False, device=device)
    states = torch.zeros(steps, state_space_size, requires_grad=False, device=device)
    actions = torch.zeros(steps, requires_grad=False, device=device)
    probs = torch.zeros(steps, requires_grad=False, device=device)
    masks = torch.zeros(steps, requires_grad=False, device=device)
    values = torch.zeros(steps, requires_grad=False, device=device)
    log_probs = torch.zeros(steps, requires_grad=False, device=device)

    state = env.reset()
    for e in range(steps):
        state = torch.FloatTensor(state).to(device)
        #printMeta(state, 'state')
        action, action_distribution = actor_critic.chooseAction(state)
        estimated_value = actor_critic.getCriticFor(state)

        next_state, reward, done, _ = env.step(action.detach().cpu().numpy())

        log_prob = action_distribution.log_prob(action)
        log_probs[e] = (log_prob)       
        masks[e] = int(not done) # Used to mask out
        rewards[e] = reward
        states[e] = state
        actions[e] = action
        
        #probs[e] = (action_distribution)
        values[e] = (estimated_value)

        state = next_state
        if done:
            env.reset()

    # We need one extra for GAE calculation
    next_state = torch.FloatTensor(next_state).to(device)
    next_value = actor_critic.getCriticFor(next_state)

    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'log_probs': log_probs, 'probs':probs, 'values':values}, next_value

def proccessExperiences(next_value, raw_experience, state_space_size, device, gamma=None, tau=None):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)
    #values = values + [next_value]
    values = torch.hstack((values, next_value))
    #printMeta(values, 'long valuees')

    # Calculate General advantage estimation and returns for each step
    gae = 0
    returns = torch.zeros(state_space_size, requires_grad=False, device=device)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] - values[step] + gamma*values[step+1]*masks[step] 
        gae = delta + gamma * tau * masks[step] * gae
        #retruns.insert(0, gae + values[step])
        returns[step] = gae + values[step]
    
    return {**raw_experience, 'returns':returns}

def miniBatchIter(mini_batch_size, experiences):
    batch_size = len(experiences[list(experiences.keys())[0]])
    for _ in range(batch_size//mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield {key: field[rand_ids] for key, field in experiences.items()}

def getDevice(force_cpu=False):
    device = None
    if force_cpu:
        return torch.device('cpu')

    targetGPU = "GeForce GTX 1080 Ti"

    if torch.cuda.is_available():
        targetDeviceNumber = None

        print("There are %d available GPUs:"%torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            prefix = "    "
            if torch.cuda.get_device_name(i) == targetGPU:
                targetDeviceNumber = i
                prefix = "[🔥]"

            print("%s %s"%(prefix, torch.cuda.get_device_name(i)))

        if targetDeviceNumber != None:
            device = torch.device('cuda:%d'%targetDeviceNumber)
        else:
            torch.device('cuda')
            raise Exception("Cannot find target GPU")
    else:
        device = torch.device('cpu')
        raise Exception("Not using GPU")

    return device

def plotScore(score, name):
    plt.plot(score)
    plt.ylabel("Score")
    plt.xlabel('Something')
    plt.savefig('%s.png'%name)

def printMeta(tensor, name):
    shape = str(tensor.shape)
    device = str(tensor.device)
    grad = str(tensor.requires_grad)
    print("%s:  %s  %s  %s"%(name, shape, device, grad))

# Hyper Parameters
discount_gamma = 0.99
GAE_tau = 0.95
epochs = 20
timesteps = 128 
mini_batch_size = 32
entropy_coeff = 0.001
vf_coeff = 1
epsilon = 0.2
episodes = 100

# Logging parameters
video_every = 800
test_interval = 10
test_batch_size = 5

env_names = {
    'gravitar': 'Gravitar-ram-v0',
    'cartpole': 'CartPole-v0',
    'spaceInvaders': 'SpaceInvaders-ram-v0',
    }

env_name = env_names['spaceInvaders']
env = gym.make(env_name)
env_test = gym.make(env_name)
#env_test = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%(video_every*test_batch_size))==0, force=True)


assert(len(env.observation_space.shape)==1)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n
print("inputs: %d   outputs: %d   for: %s"%(num_inputs, num_outputs, env_name))

device = getDevice(force_cpu=False)
model = ActorCritic(num_inputs, num_outputs, 128).to(device)


score = []

episode_iter = trange(0, episodes)
for i in episode_iter:
    raw_experience, next_value = accrueExperience(env, model, num_inputs, device, steps=timesteps)

    experience = proccessExperiences(next_value, raw_experience, num_inputs, device, gamma=discount_gamma, tau=GAE_tau)

    #returns   = torch.cat(experience['returns']).detach()
    #print(returns)
    returns   = experience['returns'].detach()
    #printMeta(returns, 'returns')
    #log_probs = torch.Tensor(experience['log_probs']).detach()
    log_probs = experience['log_probs'].detach()
    #printMeta(log_probs, 'log_probs')
    #values    = torch.Tensor(experience['values']).detach()
    values    = experience['values'].detach()
    #printMeta(values, 'values')
    #states = torch.Tensor()
    #states    = torch.cat(experience['states']).view(len(experience['states']), -1)
    states    = experience['states']
    #printMeta(states, 'states')
    #actions   = torch.Tensor(experience['actions'])
    actions   = experience['actions']
    #printMeta(actions, 'actions')
    #advantage = returns - values
    advantage = torch.Tensor.detach(returns - values)
    #printMeta(advantage, 'advantage')

    experience = {'returns':returns, 'log_probs':log_probs, 'values':values, 'states':states, 'actions':actions, 'advantage':advantage}
    model.learn(epochs, experience, 
        mini_batch_size=mini_batch_size,
        entropy_coeff=entropy_coeff,
        vf_coeff=vf_coeff,
        clip_epsilon=epsilon)

    if i % test_interval == 0:
        score_batch = [testReward(env_test, model, device) for _ in range(test_batch_size)]
        avg_score = np.mean(score_batch)
        score.append(avg_score)
        if i % (video_every*test_interval) == 0:
            plotScore(score, 'score')
        episode_iter.set_description("Current Score %.1f  " % avg_score)


plotScore(score, 'score')