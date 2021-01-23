import torch
from torch import nn
import torch.nn.functional as F

from agents.utils.ReplayMemory import Transition
from agents.BaseAgent import BaseAgent

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fan_in_uniform_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1./(fan_in**.5)
    nn.init.uniform_(tensor, -w, w)

class LinearLN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLN, self).__init__()
        self._fc    = nn.Linear(input_dim, output_dim)
        self._ln    = nn.LayerNorm(output_dim)

        fan_in_uniform_init(self._fc.weight)
        fan_in_uniform_init(self._fc.bias)
    
    def forward(self, x):
        return F.relu(self._ln(self._fc(x)))

class DDPGActor(nn.Module):

    def __init__(self, input_dim, output_dim, fc_dims = [(256, 1024), (1024, 1024)]):
        super(DDPGActor, self).__init__()

        self._fc_ln_layers = nn.ModuleList([
            LinearLN(size[0], size[1])
            for size in [(input_dim, fc_dims[0][0]) ,*fc_dims]
        ])

        self._mu = nn.Linear(fc_dims[-1][1], output_dim)
        nn.init.uniform_(self._mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self._mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, x):
        for layer in self._fc_ln_layers:
            x = F.relu(layer(x))

        return F.tanh(self._mu(x))

class DDPGCritic(nn.Module):
    
    def __init__(self, input_dim, action_space_dim, fc_dims = [(256, 1024), (1024, 1024)]):
        super(DDPGCritic, self).__init__()
        
        fc_layers = []
        for i, size in enumerate([(input_dim, fc_dims[0][0]) ,*fc_dims]):
            if i == len(fc_dims):
                fc_layers.append(
                    LinearLN(size[0] + action_space_dim, size[1])
                )
            else:
                fc_layers.append(
                    LinearLN(size[0], size[1])
                )

        self._fc_ln_layers = nn.ModuleList(fc_layers)

        self._V = nn.Linear(fc_dims[-1][1], 1)
        nn.init.uniform_(self._V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self._V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, x, actions):
        for i, layer in enumerate(self._fc_ln_layers):
            if i == len(self._fc_ln_layers)-1:
                x = torch.cat((x, actions), 1)

            x = F.relu(layer(x))

        return self._V(x)

class DDPG(BaseAgent):
    def __init__(self, in_size, action_space_size, noise=None, actor_lr=1e-4, critic_lr=1e-3, gamma=.99):
        """Creates instance of Deep Deterministic Policy Gradient(DDPG) network
           https://arxiv.org/abs/1509.02971

        Args:
            in_size (int): Size of the input
            action_space_size (int): Dimension of the action space
            noise (BaseNoise): noise to include when picking an action
            actor_lr (float, optional): Learning rate for Actor. Defaults to 1e-4.
            critic_lr (float, optional): Learning rate for Critic. Defaults to 1e-3.
            gamma (float, optional): Discount factor. Defaults to .99.
        """
        super(DDPG, self).__init__(noise)

        self._gamma         = gamma

        # Actor networks
        self.actor          = DDPGActor(input_dim=in_size, output_dim=action_space_size).to(device)
        self.actor_target   = DDPGActor(input_dim=in_size, output_dim=action_space_size).to(device)

        # Critic networks
        self.critic         = DDPGCritic(input_dim=in_size, action_space_dim=action_space_size).to(device)
        self.critic_target  = DDPGCritic(input_dim=in_size, action_space_dim=action_space_size).to(device)

        # Networks optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.actor.parameters(),lr=critic_lr)

        self._sync_nets_hard()

    def update_params(self, transitions):
        """Update parameters for both network by supplied sampled batch

        Args:
            transitions (List[Transition]): List of transitions

        Returns:
            tuple(float, float): Value loss, Policy loss
        """
        # Transpose the batch
        # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))

        # Create separate batches for data in transitions
        state_batch         = torch.cat(batch.state).to(device)
        action_batch        = torch.cat(batch.action).to(device)
        next_state_batch    = torch.cat(batch.next_state).to(device)
        reward_batch        = torch.cat(batch.reward).to(device)
        done_batch          = torch.cat(batch.done).to(device).to(torch.float)

        next_action_batch           = self.actor_target(next_state_batch)
        next_state_action_values    = self.critic_target(next_state_batch, next_action_batch.detach())

        expected_values = reward_batch.unsqueeze(1) + (1.0 - done_batch.unsqueeze(1))*self._gamma*next_state_action_values
        
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self._sync_nets_soft()

        return value_loss.item(), policy_loss.item()

    def _sync_nets_hard(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def _sync_nets_soft(self, tau=0.001):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_model(self):
        """Saves the model to predefined storage
        TODO: Implement storing files
        """
        return super().save_model()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.eval()
        self.critic_target.eval()

    @property
    def action_policy(self):
        return self.actor

    def select_action(self, state):
        return super().select_action(state)

    def forward(self, x):
        x = x.to(device)

        self.actor.eval()
        mu = self.actor(x).data
        self.actor.train()

        return mu