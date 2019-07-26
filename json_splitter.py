import json
import os

raw = open("/home/cioni/PycharmProjects/trajRNN/data/dataset_trajectories.json")
j =json.load(raw)


# for k in j:
#     print(k)
#     df=open(k,'w')
#     json.dump(j[k],df)

with open("orign","w") as out:
    # json.dump(j,out)
    out.write(str(j))