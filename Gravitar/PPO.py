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

    def learn(self, stuff):
        pass


    

# Need an optimiser for actor critic


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

    for e in range(steps):
   
        action, action_distribution = actor_critic.chooseAction(state)
        estimated_value = actor_critic.getCritic(state)

        next_state, reward, done, _ = env.step(action)
        
        masks.append(int(not done)) # Used to mask out
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        
        probs.append(action_distribution)
        values.append(estimated_value)

        state = next_state
        if done:
            env.reset()

    # Get value again? Seems pointless -- unless we get final value

    return {'masks':masks, 'rewards':rewards, 'states':states, 'actions':actions, 'probs':probs, 'values':values}

def proccessExperiences(next_value, raw_experience, gamma=0.99, tau=0.95):
    masks, rewards, values = itemgetter('masks', 'rewards', 'values' )(raw_experience)
    values += [next_value]

    # Calculate General advantage estimation and returns for each step
    gae = 0
    retruns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] - values[step] + gamma*values[step+1]*masks[step] 
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    
    return {**raw_experience, 'returns':retruns}

def miniBatchIter(mini_batch_size, mini_batch_count, experiences):
    batch_size = len(experiences[experiences.keys()[0]])
    for _ in range(mini_batch_count):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield {key: field[rand_ids, :] for key, field in experiences}


def teachActorCritic(actor_critic, experiences, epochs=10):

    for e in range(epocs):
        al, cl = actor_critic.learn(experiences)
    

