import torch
from torch import nn

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

    def chooseAction(self, state):
        # sample prob dist of state for next action
        # return dist as well
        pass

    def getCritic(self, state):
        # get critique for a certian state
        pass


    

# Need an optimiser for actor critic


def testReward(env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        # GET ACTION
        action = "TODO"

        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

def accrueExperience(env, actor_critic, frames=128):

    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    values = []

    for e in range(frames):
   
        action, action_distribution = actor_critic.chooseAction(state)
        estimated_value = actor_critic.getCritic(state)

        next_state, reward, done, _ = env.step(action)
        
        dones.append(done)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        
        probs.append(action_distribution)
        values.append(value)

        state = next_state
        if done:
            env.reset()

    # Get value again? Seems pointless -- unless we get final value

    return {'done':dones, 'rewards':rewards, 'states':states, 'actions':actions, 'probs':probs}

def proccessExperiences(raw_experience):
    dones, rewards, states, actions, probs = itemgetter('done', 'rewards', 'states', 'actions', 'probs')(raw_experience)

    pass

def teachActorCritic(experiences, epochs=10):

    pass

