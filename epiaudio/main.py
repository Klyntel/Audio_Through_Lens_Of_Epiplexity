""" Basic Experiment Outline.

Natural Langauge Experiments were conducted in Jax under the picodo folder of epiplexity. 

We need to potentially reimplement some of the work to mirror it but put the work on comet-ml instead.
"""
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv

import sys
import os

# Some magic to get utils and data in picodo in path so things reslove correctly
# TODO: I wouldn't mind if someone gives some feedback here
# This feels a bit awkward a solution, a better on may exist
# Well a better one does but we can't fork it rn sooooooo
path_to_og_work = os.path.join(
    os.path.dirname(os.path.dirname(__file__)
), "epiplexity", "picodo")
print(os.path.dirname(__file__))
sys.path.insert(0, path_to_og_work)

# Below is mostly what was in the og main.py
import epiaudio.train as train #noqa E402
# Ignore so we can do magic with imports


@hydra.main(version_base=None, config_path=os.path.join(
    path_to_og_work, 'configs'
), config_name='default')
def main(c: DictConfig):
    load_dotenv()
    train.train_and_evaluate(c)


if __name__ == '__main__':
    main()
