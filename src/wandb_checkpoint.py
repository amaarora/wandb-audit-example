"""
This script has been adapted from `pytorch-image-models` checkpoint saver script
written by Ross Wightman.
This script adds Weights and Biases artifact integration on top.
(https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/checkpoint_saver.py)
"""

""" Checkpoint Saver
Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.
Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import operator
import os
import logging
import wandb
import torch

_logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            checkpoint_prefix='checkpoint',
            checkpoint_dir='',
            decreasing=False,
            max_history=2,
            wandb_run=None):

        # wandb run
        self.wandb_run = wandb_run if wandb_run is not None else wandb.init(job_type='model-artifact')

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.save_prefix = checkpoint_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.log_artifact(filename, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            _logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                # self.delete_artifact(os.path.basename(d[0]))
                os.remove(d[0])
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def log_artifact(self, filename, save_path):
        try: 
            artifact = wandb.Artifact(filename, type='model')
            artifact.add_file(save_path)
            self.wandb_run.log_artifact(artifact)
        except Exception as e:
            _logger.error("Exception '{}' while logging wandb artifact".format(e))

    def delete_artifact(self, filename, alias='v0'):
        api = wandb.Api()
        artifact = api.artifact(f'{self.args.project}/{filename}:{alias}')
        try: 
            artifact.delete(delete_aliases=True)
        except Exception as e:
            _logger.error("Exception '{}' while deleting wandb artifact {}".format(e, filename))