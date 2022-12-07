#!/usr/bin/env python3
#coding:utf-8

import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
warnings.simplefilter('ignore')

from functools import reduce
from munch import Munch

from meldataset import build_dataloader
from optimizers import build_optimizer
from models import build_model
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from Utils.ASR.models import ASRCNN

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True #

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)

def main(config_path):
    config = yaml.safe_load(open(config_path))
    print(config)
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    ### Get configuration
    batch_size = config.get('batch_size', 2)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    dataset_configuration = config.get('dataset_configuration', None)
    stage = config.get('stage', 'star')
    fp16_run = config.get('fp16_run', False)
    training_path = config.get('training_path', "Data/training_list.txt")
    validation_path = config.get('validation_path', "Data/validation_list.txt" )
    ###
    
    # load dataloader 
    train_dataloader = build_dataloader(training_path,dataset_configuration,
                                        batch_size=batch_size,
                                        num_workers=1,
                                        device=device)
    
    val_dataloader = build_dataloader(validation_path,dataset_configuration,
                                        batch_size=batch_size,
                                        num_workers=1,
                                        device=device)

    # load pretrained ASR model, FROZEN
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    with open(ASR_config) as f:
            ASR_config = yaml.safe_load(f)
    ASR_model_config = ASR_config['model_params']
    ASR_model = ASRCNN(**ASR_model_config)
    params = torch.load(ASR_path, map_location='cpu')['model']
    ASR_model.load_state_dict(params)
    _ = ASR_model.eval()    
    
    
    # build model
    model, model_ema = build_model(Munch(config['model_params']), ASR_model)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 2e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    
    _ = [model[key].to(device) for key in model]
    _ = [model_ema[key].to(device) for key in model_ema]
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict=scheduler_params_dict)

    trainer = Trainer(args=Munch(config['loss_params']), model=model,
                            model_ema=model_ema,
                            optimizer=optimizer,
                            device=device,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            logger=logger,
                            fp16_run=fp16_run)

    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    for _ in range(1, epochs+1):
        epoch = trainer.epochs
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.4f' % (key, value))
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        trainer.save_checkpoint(osp.join(log_dir, 'ex_6_happy_backup.pth'))
        if epoch in [50,100,149]:
            trainer.save_checkpoint(osp.join(log_dir, f'ex_6_happy_{epoch}.pth'))
    return 0

if __name__=="__main__":
    main()
