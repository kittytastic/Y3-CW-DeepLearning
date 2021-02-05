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


class TransformFrame(nn.Module):
    def __init__(self):
        super(TransformFrame, self).__init__()
        

    def forward(self, x):
        x = x[:200, :160]
        #printMeta(x, 'linear')
        grey_image = x.mean(dim=2)
        return grey_image


class ConvFrames(nn.Module):
    def __init__(self, frames):
        super(ConvFrames, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=(2,2), stride=2) # -> 4 x 100 x 80

        self.conv1 = Block(frames, frames*2, 2, 0) # -> 8 x 48 x 38
        self.conv2 = Block(frames*2, frames*4, 2, 0) # -> 8 x 25 x 20
        self.conv3 = Block(frames*4, frames*8, 2, 0)

    def forward(self, x):
        x = self.downsample(x)
        #printMeta(x, 'down')
        x = self.conv1(x)
        #printMeta(x, 'con1')
        x = self.conv2(x)
        #printMeta(x, 'conv2')
        x = self.conv3(x)
        #printMeta(x, 'conv3')
        x = x.flatten(1)
        return x


class Block(nn.Module):
    def __init__(self, in_f, out_f, stride, padding):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.f(x)

class ActorCritic(nn.Module):
    def __init__(self, frame_stack, num_outputs, hidden_size, lr=3e-4):
        super(ActorCritic, self).__init__()

        self.frame_to_state=ConvFrames(frame_stack)
        #self.frames_to_state(
        #    Block(),
        #    Block(),
        #    Block(),
        #)
        
        self.first_layer_size = 792*frame_stack
        self.critic = nn.Sequential(
            nn.Linear(self.first_layer_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(self.first_layer_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=0),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
       

    def chooseAction(self, frames, p=False):
        if p:
            printMeta(frames, 'frames')
        state = self.frame_to_state(frames)
        if p:
            printMeta(state, 'state')
        mu = self.actor(state)
        dist = torch.distributions.Categorical(mu)
        action = dist.sample()
        return action, dist

    def getCriticFor(self, frames):
        state = self.frame_to_state(frames)
        return self.critic(state)

    def learn(self, epochs, experience, clip_epsilon=None, mini_batch_size=None, entropy_coeff=None, vf_coeff=None): 
        for _ in range(epochs):
            for mini_experience_batch in miniBatchIter(mini_batch_size, experience):
            
                state, action, old_log_probs, retruns, advantage = itemgetter('states', 'actions', 'log_probs', 'returns', 'advantage' )(mini_experience_batch)
                
                #printMeta(state, 'State in learn')
                _, dist = self.chooseAction(state, p=False)
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


def testReward(env, actor_critic, device, frame_stack_depth):
    total_reward = 0
    state = env.reset()
    done = False

    frame_stack = FrameStack(frame_stack_depth)
    state = torch.FloatTensor(state).to(device)
    frame_stack.setFirstFrame(state)
    
    while not done:
        state = frame_stack.asState()
        action, _ = actor_critic.chooseAction(state)
        next_frame, reward, done, _ = env.step(action.detach().cpu().numpy())
        next_frame = torch.FloatTensor(next_frame).to(device)
        frame_stack.pushFrame(next_frame)
        total_reward += reward

    return total_reward


class FrameStack():
    def __init__(self, depth):
        self.depth = depth
        #self.stack = torch.zeros((size, frame_size), device=device, requires_grad=False)
        self.stack = None
        self.frame_transformer = TransformFrame()

    def pushFrame(self, frame):
        #printMeta(frame, 'og frame')
        new_frame = self.frame_transformer(frame)
        #printMeta(new_frame, 'transformed')
        new_frame = new_frame.unsqueeze(0)
        #printMeta(new_frame, 'new frame')
        #printMeta(self.stack[1:], 'stack')
        self.stack =  torch.cat((self.stack[1:], new_frame))

    def setFirstFrame(self, frame):
        new_frame = self.frame_transformer(frame)
        self.stack = new_frame.repeat(self.depth, 1, 1)
    
    def getTrueState(self):
        return self.stack

    def asState(self):
        return self.stack.unsqueeze(0)

def accrueExperience(env, actor_critic, frame_stack_depth, device, steps=None):

    rewards = torch.zeros(steps, requires_grad=False, device=device)
    states = torch.zeros(steps, frame_stack_depth, 200, 160, requires_grad=False, device=device)
    actions = torch.zeros(steps, requires_grad=False, device=device)
    #probs = torch.zeros(steps, requires_grad=False, device=device)
    masks = torch.zeros(steps, requires_grad=False, device=device)
    values = torch.zeros(steps, requires_grad=False, device=device)
    log_probs = torch.zeros(steps, requires_grad=False, device=device)
    
    frame_stack = FrameStack(frame_stack_depth)

    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    frame_stack.setFirstFrame(state)
    for e in range(steps):
        state = frame_stack.asState()

        action, action_distribution = actor_critic.chooseAction(state)
        estimated_value = actor_critic.getCriticFor(state)

        #printMeta(state, 'in here state')
        #printMeta(estimated_value, 'in here estimated val')
        #printMeta(action, 'in here action')

        #raise Exception("Stawppp")
        next_frame, reward, done, _ = env.step(action.detach().cpu().numpy())

        log_prob = action_distribution.log_prob(action)
        log_probs[e] = log_prob     
        masks[e] = int(not done) # Used to mask out
        rewards[e] = reward
        states[e] = state
        actions[e] = action
        values[e] = estimated_value

        next_frame = torch.FloatTensor(next_frame).to(device)
        frame_stack.pushFrame(next_frame)
        
        if done:
            next_frame = env.reset()
            next_frame = torch.FloatTensor(next_frame).to(device)
            frame_stack.setFirstFrame(next_frame)


    #printMeta(states, 'states')
    #printMeta(rewards, 'rewards')
    #printMeta(actions, 'actions')
    #printMeta(masks, 'masks')
    #printMeta(values, 'values')
    #printMeta(log_probs, 'log_probs')

    # We need one extra value for GAE calculation
    next_value = actor_critic.getCriticFor(frame_stack.asState()).flatten()
    #printMeta(next_value, 'next_value')


    #return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'log_probs': log_probs, 'probs':probs, 'values':values}, next_value
    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'log_probs': log_probs, 'values':values}, next_value

def proccessExperiences(next_value, raw_experience, steps, device, gamma=None, tau=None):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)
    
    values = torch.hstack((values, next_value))
   
    scores = []
    running_score = 0
    for i in range(len(rewards)):
        running_score += rewards[i]
        if masks[i]==0.0:
            scores.append(running_score.detach().cpu().numpy())
            running_score = 0
   

    # Calculate General advantage estimation and returns for each step
    gae = 0
    returns = torch.zeros(steps, requires_grad=False, device=device)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] - values[step] + gamma*values[step+1]*masks[step] 
        gae = delta + gamma * tau * masks[step] * gae
        #retruns.insert(0, gae + values[step])
        returns[step] = gae + values[step]
    
    return {**raw_experience, 'returns':returns, 'scores': scores}

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


mini_batch_size = 32
entropy_coeff = 0.01
vf_coeff = 1
epsilon = 0.2

learning_rate = 3e-4

epochs = 5
episodes = 2
timesteps = 2048 
frame_stack_depth = 4

# Logging parameters
video_every = 1
test_interval = 10
test_batch_size = 1

env_names = {
    'gravitar': 'Gravitar-v0',
    'spaceInvaders': 'SpaceInvaders-v0',
    'breakout': 'Breakout-v0',
    }

env_name = env_names['breakout']
env = gym.make(env_name)
#env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%1)==0, force=True)
env_test = gym.make(env_name)
env_test = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%(video_every*test_batch_size))==0, force=True)


##### 792
'''
con_boi = ConvFrames(3)
fs = FrameStack(3)

frame = torch.rand(210, 160, 3)
fs.setFirstFrame(frame)
fb = fs.asState()
printMeta(fb, "States")

lots = fb.repeat((30, 1, 1, 1))
printMeta(lots, "lots")

tmp = con_boi(fb)
printMeta(tmp, "Convved")

tmp = con_boi(lots)
printMeta(tmp, "lots Convved")

exit()
'''

num_inputs  = env.observation_space
num_outputs = env.action_space.n

print(env.observation_space)
print("inputs: %s   outputs: %d   for: %s"%(num_inputs, num_outputs, env_name))
device = getDevice(force_cpu=False)

model = ActorCritic(frame_stack_depth, num_outputs, 128, lr=learning_rate).to(device)


score_over_time = []

episode_iter = trange(0, episodes)
for i in episode_iter:
    raw_experience, next_value = accrueExperience(env, model, frame_stack_depth, device, steps=timesteps)

    experience = proccessExperiences(next_value, raw_experience, timesteps, device, gamma=discount_gamma, tau=GAE_tau)
    scores = np.array(experience['scores'])
    #print(scores)
    avg_score = scores.mean()
    episode_iter.set_description("Current Score %.1f  (%d games)" % (avg_score, len(scores)))
    

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
        score_batch = [testReward(env_test, model, device, frame_stack_depth) for _ in range(test_batch_size)]
        #avg_score = np.mean(score_batch)
        #score.append(avg_score)
        if i % (video_every*test_interval) == 0:
            plotScore(scores, 'score')
        #episode_iter.set_description("Current Score %.1f  " % avg_score)
    

plotScore(scores, 'score')