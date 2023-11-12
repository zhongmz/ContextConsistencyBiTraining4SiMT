#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

from fairseq_cli.train import cli_main
import wandb
import yaml
from datetime import datetime
#wandb.login(key='5d96e5f22b7eb91f28d15c86739d6cbf22175223')
if __name__ == '__main__':  
    # with open('/home/mzzhong/wait-info/scripts/iwslt14deen/wandb.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # wandb.init(project='Wait-Info', name=config['name'])
    cli_main()
    #wandb.finish()
