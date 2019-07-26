import yaml
import utils

def test_feeder():
    with open("config.yaml") as cnf:
        cfg = yaml.load(cnf)
    loader=utils.Loader(cfg)
    x,gt=loader.serve()
    print(x)
    print(gt)
test_feeder()