import yaml
import os
import train_top_view
import shutil
def get_config(cf):
    print(cf)
    with open(cf) as raw_cfg:

        cfg = yaml.load(raw_cfg)
    return cfg

def main():
    for cfs in os.listdir("confs"):

        cfg=get_config("confs/"+cfs)
        print(cfs+str(cfg['load']))
        path=train_top_view.train_multiple(cfg)
        shutil.copy("confs/"+cfs, path + "/data/" + "config_real.yaml")
        cfg['load_path']=path+"/model/model_last.ckpt"
        cfg['test']=True
        cfg['load']=True
        train_top_view.test_multiple(cfg)

main()
