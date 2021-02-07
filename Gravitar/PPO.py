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
import torchvision

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
        x = x[20:200, :160]
        grey_image = x.mean(dim=2)
        grey_image = grey_image.unsqueeze(0)
        smaller = torchvision.transforms.functional.resize(grey_image, size=(80,80))
        return smaller.squeeze()

class FrameStack():
    def __init__(self, depth, device, skip=4):
        self.depth = depth
        self.device = device
        self.stack = None
        self.frame_transformer = TransformFrame()

        self.h = 80
        self.w = 80

        self.stack_every = skip
        self.current_frame = 0
        

    def pushFrame(self, frame):
        self.current_frame += 1
        
        if self.current_frame != self.stack_every:
            return

        self.current_frame = 0

        frame_tensor = torch.FloatTensor(frame).to(self.device)
        new_frame = self.frame_transformer(frame_tensor)
        new_frame = new_frame.unsqueeze(0)
        self.stack =  torch.cat((self.stack[1:], new_frame))

    def setFirstFrame(self, frame):
        self.current_frame = 0
        frame_tensor = torch.FloatTensor(frame).to(self.device)
        new_frame = self.frame_transformer(frame_tensor)
        self.stack = new_frame.repeat(self.depth, 1, 1)
    
    def getTrueState(self):
        return self.stack

    def asState(self):
        return self.stack.unsqueeze(0)


class Block(nn.Module):
    def __init__(self, in_f, out_f, stride, padding, kernel):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.f(x)

class ConvFrames(nn.Module):
    def __init__(self, frames):
        super(ConvFrames, self).__init__()
        # <- 4 x 80 x 80
        self.conv1 = Block(frames, 16, 4, 0, 8) # -> 8 x 39 x 39
        self.conv2 = Block(16, 32, 2, 0, 4) # -> 16 x 19 x 19
        #self.conv3 = Block(64, 128, 2, 0, 3) # -> 32 x 9 x 9

    def forward(self, x):
        x = self.conv1(x)
        #printMeta(x, 'con1')
        x = self.conv2(x)
        #printMeta(x, 'conv2')
        #x = self.conv3(x)
        #printMeta(x, 'conv3')
        x = x.flatten(1)
        return x

class ActorCritic(nn.Module):
    def __init__(self, frame_stack, num_outputs, hidden_size, lr=3e-4):
        super(ActorCritic, self).__init__()

        self.conv_output_size = neurons_per_frame*frame_stack
        
        self.critic = nn.Sequential(
            ConvFrames(frame_stack),
            nn.Linear(self.conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        self.actor = nn.Sequential(
            ConvFrames(frame_stack),
            nn.Linear(self.conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
       

    def getActionDist(self, frames):
        actor_ops = self.actor(frames)
        dist      = torch.distributions.Categorical(actor_ops)
        return dist

    def getCriticFor(self, frames):
        return self.critic(frames)

    def learn(self, epochs, experience, device, clip_epsilon=None, mini_batch_size=None, entropy_coeff=None, vf_coeff=None): 
        loss_acc = torch.zeros(1, device=device, requires_grad=False)
        entropy_acc = torch.zeros(1, device=device, requires_grad=False)
        actor_acc = torch.zeros(1, device=device, requires_grad=False)
        critic_acc = torch.zeros(1, device=device, requires_grad=False)
        
        for _ in range(epochs):
            for mini_experience_batch in miniBatchIter(mini_batch_size, experience):
            
                state, action, old_log_probs, retruns, advantage = itemgetter('states', 'actions', 'log_probs', 'returns', 'advantage' )(mini_experience_batch)
                
                dist = self.getActionDist(state)
                value = self.getCriticFor(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0-clip_epsilon, 1.0+clip_epsilon)*advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (retruns - value).pow(2).mean()

                loss =  vf_coeff * critic_loss + actor_loss - entropy_coeff * entropy

                loss_acc += loss.detach()
                critic_acc += (vf_coeff * critic_loss).detach()
                actor_acc += actor_loss.detach()
                entropy_acc -= (entropy_coeff * entropy).detach()

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
        
        loss_acc = loss_acc.mean().detach().cpu().numpy()
        entropy_acc = entropy_acc.mean().detach().cpu().numpy()
        actor_acc = actor_acc.mean().detach().cpu().numpy()
        critic_acc = critic_acc.mean().detach().cpu().numpy()

        return {'loss': loss_acc, 'entropy': entropy_acc, 'actor':actor_acc, 'critic':critic_acc}


def testReward(env, actor_critic, device, frame_stack_depth):
    total_reward = 0
    state = env.reset()
    done = False

    frame_stack = FrameStack(frame_stack_depth, device)
    frame_stack.setFirstFrame(state)
   
    while not done:
        state = frame_stack.asState()

        action_distribution = actor_critic.getActionDist(state)
        action = action_distribution.sample()

        next_frame, reward, done, _ = env.step(action.detach().cpu().numpy())
        
        total_reward += reward
        frame_stack.pushFrame(next_frame)

    return total_reward




def accrueExperience(env, actor_critic, frame_stack, partial_reward, device, steps=None):

    #frame_stack = FrameStack(frame_stack_depth, device)

    rewards = torch.zeros(steps, requires_grad=False, device=device)
    states = torch.zeros(steps, frame_stack.depth, frame_stack.h, frame_stack.w, requires_grad=False, device=device)
    actions = torch.zeros(steps, requires_grad=False, device=device)
    masks = torch.zeros(steps, requires_grad=False, device=device)
    values = torch.zeros(steps, requires_grad=False, device=device)
    log_probs = torch.zeros(steps, requires_grad=False, device=device)
    

    #state = env.reset()
    #frame_stack.setFirstFrame(state)
    episodes_scores = []
    total_reward = partial_reward
    for e in range(steps):
        state = frame_stack.asState()

        action_distribution = actor_critic.getActionDist(state)
        action = action_distribution.sample()


        estimated_value = actor_critic.getCriticFor(state)

        next_frame, reward, done, _ = env.step(action.detach().cpu().numpy())

        log_prob = action_distribution.log_prob(action)

        log_probs[e] = log_prob     
        masks[e] = int(not done) # Used to mask out
        rewards[e] = reward
        states[e] = state
        actions[e] = action
        values[e] = estimated_value

        total_reward += reward
        frame_stack.pushFrame(next_frame)
        
        #if frame_stack.current_frame == 4:
        #    plotFramestack(frame_stack.asState(), str(e))
        
        if done:
            episodes_scores.append(total_reward)
            total_reward = 0
            next_frame = env.reset()
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


    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'log_probs': log_probs, 'values':values}, next_value, frame_stack, episodes_scores, total_reward

def proccessExperiences(next_value, raw_experience, steps, device, gamma=None, tau=None):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)
    
    values = torch.hstack((values, next_value))
   
    # Calculate General advantage estimation and returns for each step
    gae = 0
    returns = torch.zeros(steps, requires_grad=False, device=device)
    for step in reversed(range(len(rewards))):
        #print("Reward: %s    Value:  %s     Value+1: %s"%(str(rewards[step]), str(values[step]), str(values[step+1])))
        #print("Mask: %s"%(str(masks[step])))
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

def accumulateLoss(all_loss, step_loss):
    for key in all_loss.keys():
        all_loss[key].append(step_loss[key])

def PlotAllLoss(loss, name):
    fig, axs = plt.subplots(len(loss), sharex=True, gridspec_kw={'hspace': 0})
    i=0
    for loss_name, loss in loss.items():
        axs[i].plot(loss)
        axs[i].set_ylabel(loss_name)
        i+=1
    plt.xlabel('Thing')
    plt.savefig('%s.png'%name)
    plt.close()


def CheckpointModel(model, checkpoint_name, loss):
    torch.save({'model':model.state_dict(), 'optimiser':model.optimiser.state_dict(), 'loss':loss}, '%s.chkpt'%checkpoint_name)

def RestoreModel(model, checkpoint_name):
    params = torch.load('Checkpoints/%s.chkpt'%checkpoint_name)
    model.load_state_dict(params['model'])
    model.optimiser.load_state_dict(params['optimiser'])
    loss = params['loss']
    return loss

def plotScore(score, name):
    plt.plot(score)
    plt.ylabel("Score")
    plt.xlabel('Something')
    plt.savefig('%s.png'%name)
    plt.close()

def printMeta(tensor, name):
    shape = str(tensor.shape)
    device = str(tensor.device)
    grad = str(tensor.requires_grad)
    print("%s:  %s  %s  %s"%(name, shape, device, grad))

def plotFramestack(tensor, name):
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    tensor = tensor.squeeze().unsqueeze(1)
    plt.imshow(torchvision.utils.make_grid(tensor, normalize=True).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.savefig('tmp/%s.png'%name)
    plt.close()


def debugTensor(tensor):
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    tensor=tensor.view(1, 1, 80, 80)
    plt.imshow(torchvision.utils.make_grid(tensor, normalize=True).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()

# Hyper Parameters
discount_gamma = 0.99
GAE_tau = 0.95


mini_batch_size = 32
entropy_coeff = 0.001
vf_coeff = 1
epsilon = 0.2

learning_rate = 3e-4

epochs = 5
episodes = 80
timesteps = 1024
frame_stack_depth = 4

# Logging parameters
video_every = 50
test_interval = 40
test_batch_size = 3


# constants
neurons_per_frame = 2048//4

env_names = {
    'gravitar': 'Gravitar-v0',
    'spaceInvaders': 'SpaceInvaders-v0',
    'breakout': 'Breakout-v0',
    }

env_name = env_names['breakout']
env = gym.make(env_name)
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0, force=True)
env_test = gym.make(env_name)


import time
'''


env.reset()
for i in range(10):
    print(i)
    env.render()
    time.sleep(1)
    env.step(1)
exit()
'''

'''
tf = TransformFrame()
fs = FrameStack(4, torch.device('cpu'))
con_boi = ConvFrames(4)
frame = env.reset()


fs.setFirstFrame(frame)
printMeta(fs.asState(), 'fs')
#for i in range(10):
#    env.render()
#    time.sleep(0.2)
#    frame, _ , _ , _ = env.step(1)
#    fs.pushFrame(frame)

#plotFramestack(fs.asState(), 'fs')

covved = con_boi(fs.asState())
printMeta(covved, 'conv out')


#tf_frame = tf(frame)
#printMeta(tf_frame, 'out frame')
#debugTensor(tf_frame)

exit()
'''

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


print("inputs: %s   outputs: %d   for: %s"%(num_inputs, num_outputs, env_name))
device = getDevice(force_cpu=False)

model = ActorCritic(frame_stack_depth, num_outputs, 256, lr=learning_rate).to(device)


score_over_time = []
experiment_scores = []

all_loss = {'loss':[], 'entropy':[], 'actor':[], 'critic':[]}

frame_stack = FrameStack(frame_stack_depth, device)
state = env.reset()
frame_stack.setFirstFrame(state)

partial_reward = 0
curr_episode = 0

episode_iter = trange(0, episodes)
for i in episode_iter:

    raw_experience, next_value, frame_stack, scores, partial_reward = accrueExperience(env, model, frame_stack, partial_reward, device, steps=timesteps)

    experience = proccessExperiences(next_value, raw_experience, timesteps, device, gamma=discount_gamma, tau=GAE_tau)
    

    curr_episode += len(scores)
    if len(scores) > 0:
        avg_score = np.mean(np.array(scores))
        episode_iter.set_description("Avg Score %.1f  (%d games)  %d episodes total" % (avg_score, len(scores), curr_episode))
        score_over_time += scores

    returns   = experience['returns'].detach()
    #printMeta(returns, 'returns')
    log_probs = experience['log_probs'].detach()
    #printMeta(log_probs, 'log_probs')
    values    = experience['values'].detach()
    #printMeta(values, 'values')
    states    = experience['states']
    #printMeta(states, 'states')
    actions   = experience['actions']
    #printMeta(actions, 'actions')
    advantage = torch.Tensor.detach(returns - values)
    #printMeta(advantage, 'advantage')
    
    #print(values)

    #print(experience['rewards'])

    #print(returns)

    #print(advantage)

    experience = {'returns':returns, 'log_probs':log_probs, 'values':values, 'states':states, 'actions':actions, 'advantage':advantage}
    losses = model.learn(epochs, experience, device,
        mini_batch_size=mini_batch_size,
        entropy_coeff=entropy_coeff,
        vf_coeff=vf_coeff,
        clip_epsilon=epsilon)

    accumulateLoss(all_loss, losses)
    
    if i % test_interval == 0:
        score_batch = [testReward(env_test, model, device, frame_stack_depth) for _ in range(test_batch_size)]
        avg_score = np.mean(score_batch)
        experiment_scores.append(avg_score)
        
        PlotAllLoss(all_loss, 'loss')
        plotScore(score_over_time, 'score')
        plotScore(experiment_scores, 'experiment_score')
    

PlotAllLoss(all_loss, 'loss')
plotScore(score_over_time, 'score')
plotScore(experiment_scores, 'experiment_score')

#CheckpointModel(actor_critic, 'test', all_loss)