#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
"""
##------------#
import sys
import torch
from data.experiments_img import run_and_save_experiments_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main_experiment.py <path_to_config>"))
config_path = sys.argv[1]

run_and_save_experiments_img(device, config_path)
