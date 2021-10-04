import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super().__init__(logdir)

    def add_scalars(self, loss_dict, iteration):
        for key, value in loss_dict.items():
            self.add_scalar(key, value, iteration)

    def add_images(self, tag, images, iteration):
        image = make_grid(images)
        self.add_image(tag, image, iteration)


def prepare_logger(logdir=''):
    return Logger(logdir)
    