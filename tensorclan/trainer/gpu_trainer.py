from pathlib import Path

import gc

from tensorclan.trainer.base_trainer import BaseTrainer, optimizer_to, scheduler_to
from tensorclan.utils import setup_logger
import tensorclan.dataset.zoo as tc_dataset

import torch
import torch.optim as optim
import torch.utils as utils

logger = setup_logger(__name__)


class GPUTrainer(BaseTrainer):
    r"""
    GPUTrainer: Trains the tensorclan model on GPU
    see :class:`~tensorclan.trainer.BaseTrainer` for args
    Examples:
        >>> gpu_trainer = GPUTrainer(model, loss_fn, optimizer, cfg, train_subset, test_subset, state_dict=state_dict)
        >>> gpu_trainer.start_train()
    """

    def __init__(self, *args, **kwargs):
        super(GPUTrainer, self).__init__(*args, **kwargs)

        cfg = self.config

        # set the device to GPU:0 // we don't support multiple GPUs for now
        self.device = torch.device("cuda:0")

        # self.writer.add_graph(self.model, (torch.randn(1, 6, 96, 96)))
        # self.writer.flush()

        self.model = self.model.to(self.device)

        optimizer_to(self.optimizer, self.device)
        scheduler_to(self.lr_scheduler, self.device)

    def train_epoch(self, epoch):
        r"""trains the model for one epoch
        Args:
            epoch: the epoch number
        Returns:
            Dict: loss, accuracy
        """
        logger.info(f'=> Training Epoch {epoch}')

        # clear the cache before training this epoch
        gc.collect()
        torch.cuda.empty_cache()

        # TODO: tqdm messes up colab notebook, use something else
        # pbar = tqdm(self.train_loader, dynamic_ncols=True)
        pbar = self.train_loader

        # set the model to training mode
        self.model.train()

        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # move the data of the specific dataset to our `device`
            # data = getattr(tc_dataset, self.config['dataset']['name']).apply_on_batch(
            #     data,
            #     lambda D: D.to(self.device)
            # )
            data, target = data.to(self.device), target.to(self.device)

            # zero out the gradients, we don't want to accumulate them
            self.optimizer.zero_grad()

            outputs = self.model(data.unsqueeze(1))

            # calculate the loss
            loss = self.loss_fn(outputs, target)

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                running_loss += loss.item()

            # update the gradients
            loss.backward()

            # step the optimizer
            self.optimizer.step()

            # step the scheduler
            if isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()

            # pbar.set_description(
            #     desc=f'loss={loss.item():.4f} seg_loss={l1.item():.4f} depth_loss={l2.item():.4f} batch_id={batch_idx}')

            self.writer.add_scalar(
                'BatchLoss/Train/loss', loss.item(), epoch*len(pbar) + batch_idx)

        running_loss /= len(pbar)
        accuracy = 100 * correct / total

        logger.info(f'loss: {running_loss}, accuracy: {accuracy}')

        self.writer.flush()

        return {'loss': running_loss, 'accuracy': accuracy}

    def test_epoch(self, epoch):
        r"""
        tests the model for one epoch

        Args:
            epoch: the epoch number
        Returns:
            Dict: loss, accuracy
        """
        logger.info(f'=> Testing Epoch {epoch}')

        # clear the cache before testing this epoch
        gc.collect()
        torch.cuda.empty_cache()

        # set the model in eval mode
        self.model.eval()

        # metrics and losses
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        # tqdm writes a lot of data into a single cell in colab that caushes high local browser
        # ram uses, so chuck tqdm, find some alternative ?
        # pbar = tqdm(self.test_loader, dynamic_ncols=True)
        pbar = self.test_loader

        for batch_idx, (data, target) in enumerate(pbar):
            # move the data of the specific dataset to our `device`
            # data = getattr(tc_dataset, self.config['dataset']['name']).apply_on_batch(
            #     data,
            #     lambda x: x.to(self.device)
            # )
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                outputs = self.model(data.unsqueeze(1))

                loss = self.loss_fn(outputs, target)

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                running_loss += loss.item()

            # pbar.set_description(desc=f'testing batch_id={batch_idx}')

        running_loss /= len(pbar)
        accuracy = 100 * correct / total

        logger.info(f'loss: {running_loss} accuracy: {accuracy}')

        return {'loss': running_loss, 'accuracy': accuracy}

    def start_train(self):
        r"""trains the model for self.epochs times
        the model and training state is saved at every epoch
        summary is flushed to disk every epoch
        """
        logger.info('=> Training Started')
        logger.info(f'Training the model for {self.epochs} epochs')

        for epoch in range(self.start_epoch, self.epochs):
            if self.lr_scheduler:
                lr_value = [group['lr']
                            for group in self.optimizer.param_groups][0]

                logger.info(f'=> LR was set to {lr_value}')
                self.writer.add_scalar('LR/lr_value', lr_value, epoch)

            # train this epoch
            train_metric = self.train_epoch(epoch)

            # train metrics
            self.writer.add_scalar(
                'EpochLoss/Train/loss', train_metric['loss'], epoch)
            self.writer.add_scalar(
                'EpochAccuracy/Train/accuracy', train_metric['accuracy'], epoch)

            # test this epoch
            test_metric = self.test_epoch(epoch)

            # test metrics
            self.writer.add_scalar(
                'EpochLoss/Test/loss', test_metric['loss'], epoch)
            self.writer.add_scalar(
                'EpochAccuracy/Test/accuracy', test_metric['accuracy'], epoch)

            # test_images = getattr(vdata_loader, self.config['dataset']['name']).plot_results(
            #     test_metric['results'])

            # self.writer.add_figure(
            #     'ModelImages/TestImages', test_images, epoch)

            # make sure to flush the data to the `SummaryWriter` file
            self.writer.flush()

            # check if we improved accuracy and save the model
            if self.config['use_checkpoints']:
                if test_metric['accuracy'] >= self.best_accuracy['accuracy']:

                    self.best_accuracy['accuracy'] = test_metric['accuracy']

                    logger.info('=> Accuracy improved, saving best checkpoint ...')

                    chkpt_path = Path(self.config['chkpt_dir'])
                    chkpt_path.mkdir(parents=True, exist_ok=True)

                    model_checkpoint = chkpt_path / 'model_checkpoint_best.pt'
                    train_checkpoint = chkpt_path / 'train_checkpoint_best.pt'
                    torch.save(self.model.state_dict(), model_checkpoint)

                    torch.save({
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.lr_scheduler.state_dict(),
                        'best_accuracy': self.best_accuracy,
                        'save_epoch': epoch,
                        'total_epochs': self.epochs
                    }, train_checkpoint)

                logger.info('=> Saving checkpoint ...')

                chkpt_path = Path(self.config['chkpt_dir'])
                chkpt_path.mkdir(parents=True, exist_ok=True)

                model_checkpoint = chkpt_path / 'model_checkpoint.pt'
                train_checkpoint = chkpt_path / 'train_checkpoint.pt'
                torch.save(self.model.state_dict(), model_checkpoint)

                torch.save({
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.lr_scheduler.state_dict(),
                    'best_accuracy': self.best_accuracy,
                    'save_epoch': epoch,
                    'total_epochs': self.epochs
                }, train_checkpoint)