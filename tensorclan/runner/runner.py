from typing import Dict, Any

import pprint
from pathlib import Path

import torch

from tensorclan.utils import setup_logger, get_instance_v2
from tensorclan.dataset import BaseDataset
from tensorclan.trainer import BaseTrainer
import tensorclan.dataset.zoo as tc_dataset
import tensorclan.dataset.augmentation as tc_augmentation
import tensorclan.model.zoo as tc_model
import tensorclan.trainer as tc_trainer

logger = setup_logger(__name__)


class Runner:
    config: Dict[str, Any]
    trainer: BaseTrainer

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info('=> Now simply setup_train and then start_train your model')

    def setup_train(self):
        cfg: Dict[str, Any] = self.config

        logger.info('=> Config')

        # print the config
        for line in pprint.pformat(cfg).splitlines():
            logger.info(line)


        # dataset:
        #   name: MNIST
        #   transforms: MNISTTransforms
        #   root: data
        #   loader_args:
        #       batch_size: 128
        #       num_workers: 2
        #       shuffle: True
        #       pin_memory: True

        # build transforms
        transforms = get_instance_v2(
            module=tc_augmentation,
            ctor_name=cfg['dataset']['transforms']
        )

        # build the dataset
        dataset: BaseDataset = get_instance_v2(
            module=tc_dataset,
            ctor_name=cfg['dataset']['name'],
            root=cfg['dataset']['root'],
            transforms=transforms
        )

        # get the train-test splits
        train_subset, test_subset = dataset.split_dataset()

        if cfg['use_checkpoints']:
            # check if the train_subset and test_subset indices are present in disk
            Path(cfg['chkpt_dir']).mkdir(parents=True, exist_ok=True)
            subset_file = Path(cfg['chkpt_dir']) / 'subset.pt'

            if subset_file.exists():
                # load the subset state
                logger.info('=> Found subset.pt loading indices')
                subset_state = torch.load(subset_file)

                train_subset.indices = subset_state['train_indices']
                test_subset.indices = subset_state['test_indices']
            else:
                # save the subset dict
                torch.save({'train_indices': train_subset.indices,
                            'test_indices': test_subset.indices}, subset_file)
                logger.info('=> Saved subset.pt (train, test indices)')

        # model:
        #   name: MNISTV1

        # create the model
        model = get_instance_v2(
            module=tc_model,
            ctor_name=cfg['model']['name']
        )

        # optimizer:
        #   type: AdamW
        #   args:
        #       lr: 0.01

        # create the optimizer
        optimizer = get_instance_v2(
            module=torch.optim,
            ctor_name=cfg['optimizer']['type'],
            params=model.parameters(),
            **cfg['optimizer']['args']
        )

        # loss: CrossEntropyLoss
        loss_fn = get_instance_v2(
            module=torch.nn,
            ctor_name=cfg['loss']
        )

        # check if the model init weights are specified
        # model_init: "models/model.pt"
        if 'model_init' in cfg:
            model_init = Path(cfg['model_init'])
            if model_init.exists():
                logger.info('=> Found Model init weights')
                model_state_dict = torch.load(model_init)
                model.load_state_dict(model_state_dict)

        state_dict = None
        if cfg['use_checkpoints']:

            # load the last checkpoint
            # chkpt_dir: checkpoint
            model_checkpoint = Path(cfg['chkpt_dir']) / 'model_checkpoint.pt'
            train_checkpoint = Path(cfg['chkpt_dir']) / 'train_checkpoint.pt'

            if model_checkpoint.exists():
                logger.info('=> Found model checkpoint')
                model_state_dict = torch.load(model_checkpoint)
                model.load_state_dict(model_state_dict)

            if train_checkpoint.exists():
                logger.info('=> Found train checkpoint')
                checkpoint_state = torch.load(train_checkpoint)

                optimizer.load_state_dict(checkpoint_state['optimizer'])
                save_epoch = checkpoint_state['save_epoch']
                total_epochs = checkpoint_state['total_epochs']
                logger.info(f'Start Epoch should be {save_epoch}+1')

                state_dict = checkpoint_state

            else:
                logger.info('=> No saved checkpoints found')

        if cfg['device'] == 'GPU':
            self.trainer = get_instance_v2(
                module=tc_trainer,
                ctor_name='GPUTrainer',
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                config=cfg,
                train_subset=train_subset,
                test_subset=test_subset,
                state_dict=state_dict
            )
        else:
            logger.error(f"Unsupported Device: {cfg['device']}")
            raise NotImplementedError

    def start_train(self):
        self.trainer.start_train()
