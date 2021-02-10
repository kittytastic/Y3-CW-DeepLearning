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
        grey_image = grey_image/255.0
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
        self.conv1 = Block(frames, 16, 4, 0, 8) 
        self.conv2 = Block(16, 32, 2, 0, 4) 
        self.conv3 = Block(32, 32, 1, 1, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
        probs = self.actor(frames)
        dist      = torch.distributions.Categorical(probs)
        return dist

    def getCriticFor(self, frames):
        return self.critic(frames)

    def learn(self, epochs, experience, device, clip_epsilon=None, mini_batch_size=None, entropy_coeff=None, vf_coeff=None, plot_grad=False): 

        self.train()

        loss_acc = torch.zeros(1, device=device, requires_grad=False)
        entropy_acc = torch.zeros(1, device=device, requires_grad=False)
        actor_acc = torch.zeros(1, device=device, requires_grad=False)
        critic_acc = torch.zeros(1, device=device, requires_grad=False)
        
        for _ in range(epochs):
            for mini_experience_batch in miniBatchIter(mini_batch_size, experience):
            
                state, action, old_log_probs, returns, advantage = itemgetter('states', 'actions', 'log_probs', 'returns', 'advantage' )(mini_experience_batch)

                #print("state: %s"%state)
                #printMeta(state, "State")
                #print("action: %s"%action)
                #print("old_log_probs: %s"%(old_log_probs))
                #print("returns: %s"%returns)
                #print("advantage: %s"%advantage)


                dist = self.getActionDist(state)
                #print(dist)

                value = self.getCriticFor(state).squeeze()
                #print("value: %s"%value)
                
                #print("Entopy dist: %s"%dist.entropy())
                entropy = dist.entropy().mean()
                #print("Entropy: %s"%entropy)

                new_log_probs = dist.log_prob(action)
                #print("new_log_probs: %s"%new_log_probs)
                
                ratio = (new_log_probs - old_log_probs).exp()
                #print("ratio: %s"%ratio)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0-clip_epsilon, 1.0+clip_epsilon)*advantage
                #print("surr1: %s"%surr1)
                #print("surr2: %s"%surr2)

                #print("Returns - values: %s"%str(returns - value))
                #print("Min: %s"%torch.min(surr1, surr2))

                
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns - value).pow(2).mean()

                #print("actor_loss: %s"%actor_loss)
                #print("critic_loss: %s"%critic_loss)

                loss =  vf_coeff * critic_loss + actor_loss - entropy_coeff * entropy

                self.optimiser.zero_grad()
                loss.backward()
                if plot_grad:
                    plot_grad_flow(self.named_parameters())
                self.optimiser.step()

                loss_acc += loss.detach()
                critic_acc += (vf_coeff * critic_loss).detach()
                actor_acc += actor_loss.detach()
                entropy_acc -= (entropy_coeff * entropy).detach()

        loss_acc = loss_acc.mean().detach().cpu().numpy()
        entropy_acc = entropy_acc.mean().detach().cpu().numpy()
        actor_acc = actor_acc.mean().detach().cpu().numpy()
        critic_acc = critic_acc.mean().detach().cpu().numpy()

        return {'loss': loss_acc, 'entropy': entropy_acc, 'actor':actor_acc, 'critic':critic_acc}


def testReward(env, actor_critic, device, frame_stack_depth):
    actor_critic.eval()

    total_reward = 0
    state = env.reset()
    done = False

    frame_stack = FrameStack(frame_stack_depth, device)
    frame_stack.setFirstFrame(state)
   
    while not done:
        state = frame_stack.asState()

        action_distribution = actor_critic.getActionDist(state)
        action = action_distribution.sample()

        next_frame, reward, done = playFrames(env, action.detach().cpu().numpy(), 4)
        
        total_reward += reward
        frame_stack.pushFrame(next_frame)

    return total_reward


def playFrames(env, action, frame_count):
    i = 0
    done = False
    total_reward = 0
    next_frame = None
    while not done and i<frame_count:
        i+=1

        next_frame, reward, done, _ = env.step(action)
        total_reward += reward

    return next_frame, total_reward, done


def accrueExperience(env, actor_critic, frame_stack, partial_reward, device, steps=None):

    actor_critic.eval()

    rewards = torch.zeros(steps, requires_grad=False, device=device)
    states = torch.zeros(steps, frame_stack.depth, frame_stack.h, frame_stack.w, requires_grad=False, device=device)
    actions = torch.zeros(steps, requires_grad=False, device=device)
    masks = torch.zeros(steps, requires_grad=False, device=device)
    values = torch.zeros(steps, requires_grad=False, device=device)
    log_probs = torch.zeros(steps, requires_grad=False, device=device)

    
    episodes_scores = []
    total_reward = partial_reward
    for e in range(steps):
        state = frame_stack.asState()
        action_distribution = actor_critic.getActionDist(state)
        action = action_distribution.sample()
       
        estimated_value = actor_critic.getCriticFor(state).squeeze()
       
        next_frame, reward, done = playFrames(env, action.detach().cpu().numpy()[0], 4)

        log_prob = action_distribution.log_prob(action)
       
        log_probs[e] = log_prob     
        masks[e] = int(not done) # Used to mask out
        rewards[e] = reward
        states[e] = state
        actions[e] = action
        values[e] = estimated_value

        total_reward += reward
        frame_stack.pushFrame(next_frame)
         
        if done:
            episodes_scores.append(total_reward)
            total_reward = 0
            next_frame = env.reset()
            frame_stack.setFirstFrame(next_frame)


    # We need one extra value for GAE calculation
    next_value = actor_critic.getCriticFor(frame_stack.asState()).flatten()


    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'log_probs': log_probs, 'values':values}, next_value, frame_stack, episodes_scores, total_reward

def proccessExperiences(next_value, raw_experience, steps, device, gamma=None, tau=None):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)

    values = torch.hstack((values, next_value))

    # Calculate General advantage estimation and returns for each step
    gae = 0
    returns = torch.zeros(steps, requires_grad=False, device=device)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] - values[step] + gamma*values[step+1]*masks[step] 
        gae = delta + gamma * tau * masks[step] * gae    
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

    targetGPU = ""

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
            device = torch.device('cuda')
            print("Cannot find target GPU, using cuda:0")
            #raise Exception("Cannot find target GPU")
    else:
        device = torch.device('cpu')
        print("CUDA not enabled, using CPU")
        #raise Exception("Not using GPU")

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
    plt.xlabel('Rounds')
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
    plt.xlabel('Episodes')
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

# Gradient investigating code from: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8, no license specified
from matplotlib.lines import Line2D
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    
    #plt.tight_layout()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def debugTensor(tensor):
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    tensor=tensor.view(1, 1, 80, 80)
    plt.imshow(torchvision.utils.make_grid(tensor, normalize=True).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()

def assertShape(tensor, dim):
    assert(len(tensor.shape) == dim)

# Hyper Parameters
discount_gamma = 0.99
GAE_tau = 0.95

entropy_coeff = 0.01
entropy_aneal_target = 0.01 
entropy_aneal_rounds = 150 
vf_coeff = 1
epsilon = 0.1

learning_rate = 3e-4

epochs = 3
mini_batch_size = 256
rounds = 150*2
timesteps = 1024
frame_stack_depth = 4

# Logging parameters
video_every = 15 # per episode
test_interval = 50 # per rounds
test_batch_size = 3 
gradient_plot = 15 # per rounds
marking_mode = False

# Constants
neurons_per_frame = 2048//4

env_names = {
    'gravitar': 'Gravitar-v0',
    'spaceInvaders': 'SpaceInvaders-v0',
    'breakout': 'Breakout-v0',
    }

env_name = env_names['gravitar']
env = gym.make(env_name)
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0, force=True)
env_test = gym.make(env_name)


num_inputs  = env.observation_space
num_outputs = env.action_space.n

# MARKING
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)


print("inputs: %s   outputs: %d   for: %s"%(num_inputs, num_outputs, env_name))
device = getDevice(force_cpu=False)

model = ActorCritic(frame_stack_depth, num_outputs, 256, lr=learning_rate).to(device)


score_over_time = []
experiment_scores = []

all_loss = {'loss':[], 'entropy':[], 'actor':[], 'critic':[]}

frame_stack = FrameStack(frame_stack_depth, device, skip=1)
state = env.reset()
frame_stack.setFirstFrame(state)

partial_reward = 0
curr_episode = 0

if not marking_mode:
    round_iter = trange(0, rounds)
else:
    round_iter = range(0, rounds)

best_score = 0
best_score_round = -1 

# MARKING
n_episode = 0
marking  = []

for i in round_iter:

    raw_experience, next_value, frame_stack, scores, partial_reward = accrueExperience(env, model, frame_stack, partial_reward, device, steps=timesteps)

    experience = proccessExperiences(next_value, raw_experience, timesteps, device, gamma=discount_gamma, tau=GAE_tau)
    

    curr_episode += len(scores)
    if len(scores) > 0:
        avg_score = np.mean(np.array(scores))
        if not marking_mode:
            round_iter.set_description("Avg Score %.1f  (%d games)  %d episodes total" % (avg_score, len(scores), curr_episode))
        score_over_time += scores

    if marking_mode:
        # Because I loop over rounds rather than episodes
        # Unwrap episode scores for marking script 
        for score in scores:
            
            

            if score>best_score:
                best_score = score
                best_score_round=n_episode

            if n_episode%video_every==0:
                print("round: %d, episode: %d, score: %d, best score: %d (%d episode)"%(i, n_episode, score, best_score, best_score_round))
            
            # do not change lines 44-48 here, they are for marking the submission log
            marking.append(score)
            if n_episode%100 == 0:
                print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
                    n_episode, score, np.array(marking).mean(), np.array(marking).std()))
                marking = []

            n_episode += 1

    returns   = experience['returns'].detach()
    log_probs = experience['log_probs'].detach()
    values    = experience['values'].detach()
    states    = experience['states']
    actions   = experience['actions']
    advantage = torch.Tensor.detach(returns - values)
    
    #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
    

   
    experience = {'returns':returns, 'log_probs':log_probs, 'values':values, 'states':states, 'actions':actions, 'advantage':advantage}

    current_aneal = entropy_coeff - ((entropy_coeff - entropy_aneal_target)*min((i/entropy_aneal_rounds),1))
    

    losses = model.learn(epochs, experience, device,
        mini_batch_size=mini_batch_size,
        entropy_coeff=current_aneal,
        vf_coeff=vf_coeff,
        clip_epsilon=epsilon,
        plot_grad=(i%gradient_plot==0))

    if i % gradient_plot == 0:
        plt.savefig("tmp/grad%d.png"%i, bbox_inches='tight')
        plt.close()


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