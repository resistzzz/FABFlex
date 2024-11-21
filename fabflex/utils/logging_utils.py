import os
from accelerate.logging import get_logger
import logging


class Logger(object):
    def __init__(self, accelerator, log_path):
        self.logger = get_logger('Main')
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s', ""))
        self.logger.logger.addHandler(handler)
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

    def log_stats(self, stats, epoch, args, prefix='', pretrain=True):
        if pretrain:
            msg_start = f'[{prefix}] Epoch {epoch} out of {args.protein_epochs}' + ' | '
        else:
            msg_start = f'[{prefix}] Epoch {epoch} out of {args.joint_epochs}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.5f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)
