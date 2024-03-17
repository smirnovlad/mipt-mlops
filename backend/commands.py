import fire

from processing.ml.train import start_training
from processing.ml.infer import inference


def train(msg):
    start_training(message=msg)


def infer(msg):
    inference(message=msg)


if __name__ == '__main__':
    fire.Fire()
