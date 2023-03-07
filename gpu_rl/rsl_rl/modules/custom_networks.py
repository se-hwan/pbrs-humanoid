import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import copy

class CustomActor (nn.Module):
    def __init__(self, custom_args, num_inputs, num_actions):
        super().__init__()
        self.custom_args = custom_args
        self.actor_rom = copy.deepcopy(custom_args['actor_network'])
        if self.custom_args['freeze']:
            for param in self.actor_rom.parameters():
                param.requires_grad = False
        if self.custom_args['residual_network']:
            activation = get_activation(self.custom_args['activation'])
            layers = self.custom_args['actor_hidden_dims']
            actor_layers = []
            actor_layers.append(nn.Linear(num_inputs, layers[0]))
            actor_layers.append(activation)
            for l in range(len(layers)):
                if l == len(layers) - 1:
                    actor_layers.append(nn.Linear(layers[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(layers[l], layers[l + 1]))
                    actor_layers.append(activation)
            self.actor_res = nn.Sequential(*actor_layers)
        
    def forward(self, x): # need to specify subset of x that is for ROM - sehwan
        if self.custom_args['residual_network']:
            return self.actor_rom(x) + self.actor_res(x)
        else:
            return self.actor_rom(x)

class ResidualCritic (nn.Module):
    def __init__(self, custom_args, num_inputs):
        super().__init__()
        self.critic_rom = copy.deepcopy(custom_args['critic_network'])
        self.rom_scale = custom_args['scaling']
        self.res_network = custom_args['residual_network']
        self.idx_rom_obs = custom_args['idx_rom_obs']
        self.critic_res = copy.deepcopy(custom_args['critic_network'])
        if custom_args['freeze']:
            for param in self.critic_rom.parameters():
                param.requires_grad = False
        if self.res_network:
            activation = get_activation(custom_args['activation'])
            layers = custom_args['critic_hidden_dims']
            critic_layers = []
            critic_layers.append(nn.Linear(num_inputs, layers[0]))
            critic_layers.append(activation)
            for l in range(len(layers)):
                if l == len(layers) - 1:
                    critic_layers.append(nn.Linear(layers[l], 1))
                else:
                    critic_layers.append(nn.Linear(layers[l], layers[l + 1]))
                    critic_layers.append(activation)
            self.critic_res = nn.Sequential(*critic_layers)
    def forward(self, x): # need to specify subset of x that is for ROM - sehwan
        rom_scale = self.rom_scale
        rom_input = x[:, self.idx_rom_obs]
        out = rom_scale*self.critic_rom(rom_input)
        if self.res_network:
            resid_input = x
            out_rom = rom_scale*self.critic_rom(rom_input)
            out_res = self.critic_res(resid_input)
            out = out_rom + out_res
        return out

class ExpertCritic (nn.Module):
    def __init__(self, custom_args, num_inputs):
        super().__init__()
        self.critic_rom = copy.deepcopy(custom_args['critic_network'])
        self.rom_scale = custom_args['scaling']
        self.idx_rom_obs = custom_args['idx_rom_obs']
        if custom_args['freeze']:
            for param in self.critic_rom.parameters():
                param.requires_grad = False
    def forward(self, x): # need to specify subset of x that is for ROM - sehwan
        rom_scale = self.rom_scale
        rom_input = x[:, self.idx_rom_obs]
        out = rom_scale*self.critic_rom(rom_input)
        return out

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None