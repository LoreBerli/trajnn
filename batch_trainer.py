import yaml
import os
import train_top_view
def get_config(cf):
    with open(cf) as raw_cfg:
        cfg = yaml.load(raw_cfg)
    return cfg

def main():
    for cfs in os.listdir("confs"):
        cfg=get_config("confs/"+cfs)
        path=train_top_view.train_multiple(cfg)
        cfg['load_path']=path+"/model/model.ckpt"
        cfg['test']=True
        cfg['load']=True
        train_top_view.test_multiple(cfg)

main()