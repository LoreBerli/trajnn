import utils
import yaml


def test():
    with open("config.yaml") as config_raw:
        cfg=yaml.load(config_raw)
    ###LOADER spawna feeders
    loader=utils.Loader(cfg)

test()