# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, task_registry, wandb_helper
from gpugym import LEGGED_GYM_ROOT_DIR
import torch

import wandb

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f'{args.task}'

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    # Check if we specified that we want to use wandb
    do_wandb = train_cfg.do_wandb if hasattr(train_cfg, 'do_wandb') else False
    # Do the logging only if wandb requirements have been fully specified
    do_wandb = do_wandb and None not in (args.wandb_project, args.wandb_entity)

    if do_wandb:
        wandb.config = {}

        if hasattr(train_cfg, 'wandb'):
            what_to_log = train_cfg.wandb.what_to_log
            wandb_helper.craft_log_config(env_cfg, train_cfg, wandb.config, what_to_log)

        print(f'Received WandB project name: {args.wandb_project}\nReceived WandB entitiy name: {args.wandb_entity}\n')
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   group=args.wandb_group,
                   config=wandb.config,
                   name=experiment_name)

        ppo_runner.configure_wandb(wandb)
        ppo_runner.configure_learn(train_cfg.runner.max_iterations, True)
        ppo_runner.learn()

        wandb.finish()
    else:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
