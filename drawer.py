from PIL import Image,ImageDraw
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def back_transform(tredpoints):
        focalLength = 721.0
        centerX = 1242/2.0
        centerY = 374/2.0
        twod_points=[]

        for pt in tredpoints:
            x=centerX+(pt[0]*focalLength)/pt[2]
            y=centerY+(pt[1]*focalLength)/pt[2]
            twod_points.append([x,y])
        #print(np.max(tredpoints[:,2]))
        #print(np.min(tredpoints[:,2]))
        return np.array(twod_points),tredpoints[:,2]

def draw_points(past,pts,gt, cfg,path_orig,box):

    sx,sy=path_orig[3],path_orig[4]


    if(not cfg["old"]):
        box[0]=(box[0]+1.0)*(sx/2.0)
        box[1]=(box[1]+0.5)*(sy)
        box[2]=(box[2]+1.0)*(sx/2.0)
        box[3]=(box[3]+0.5)*(sy)
    else:
        box[0]=(box[0])*(sx/4.0)
        box[1]=(box[1])*(sy)
        box[2]=(box[2])*(sx/4.0)
        box[3]=(box[3])*(sy)
    new_box=box
    #new_box=new_box*np.array([sx/2.0,sy])
    orig = Image.open(cfg['img_path']+"/"+path_orig[0].split("/")[-2]+"/"+path_orig[0].split("/")[-1].zfill(4)+"/"+path_orig[1].replace("frame_","") + ".png")
    #orig=Image.open("/home/cioni/data/image_02/"+path_orig[0].replace("video_","")+"/"+path_orig[1].replace("frame_","")+".png")
    orig=orig.resize(((int(sx),int(sy))))
    _out=orig
    #_out = Image.new("RGB",(cfg['out_size_x'],cfg['out_size_y']),color=(255,255,255))
    drawer=ImageDraw.Draw(_out)
    #gt = gt + 1.0
    #(float(i[3] / 2.0), float(i[4]))
    if(cfg['center']):
        poins=gt+np.array([1.0,0.5])
    else:
        poins=gt
    poins = poins * (sx/2.0, sy)



    # poins=np.cumsum(poins,0)
    #poins=np.insert(poins,0,[0,0],0)
    #poins,zs=back_transform(gt)

    poins = np.array(poins, dtype=np.int32)

    lt = [tuple(p) for p in poins.tolist()]
    for i,l in enumerate(lt):
        r = min(255, 140 + i * 10)
        sz=2#int(np.clip(zs[i]*100.0,0,10))
        drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] + sz, l[1] + sz)], fill=(0, 0, r))
    drawer.point(lt, fill=(0, 0, 128))
    #past = past + 1.0

    if(cfg['center']):
        poins=pts+np.array([1.0,0.5])
    else:
        poins=pts
    poins = poins  *(sx/2.0, sy)

    # poins=np.cumsum(poins,0)
    #poins=np.insert(poins,0,[0,0],0)
    #poins,zs=back_transform(pts)
    poins=np.array(poins,dtype=np.int32)
    #poins=backtransform(poins)
    lt=[tuple(p) for p in poins.tolist()]
    for i,l in enumerate(lt):
        r=min(255,160+i*10)
        sz=2#int(np.clip(zs[i]*100.0,0,10))
        drawer.rectangle([(l[0]-sz,l[1]-sz),(l[0]+sz,l[1]+sz)],fill=(r,0,0))
    drawer.point(lt,fill=(128,0,0))

    if(cfg['center']):
        poins=past+np.array([1.0,0.5])
    else:
        poins=past
    poins= poins * (sx/2.0, sy)

    # poins=np.cumsum(poins,0)
    #poins=np.insert(poins,0,[0,0],0)
    #poins=np.array(poins,dtype=np.int32)

    #poins,zs=back_transform(past)

    points=np.array(poins,dtype=np.int32)
    lt=[tuple(p) for p in poins.tolist()]
    for i,l in enumerate(lt):
        if(i<cfg['prev_leng']):
            r = min(255, 90 + i * 10)
        else:
            r = min(255, 160 + i * 10)
        sz=2#int(np.clip(zs[i]*100.0,0,10))
        drawer.rectangle([(l[0]-sz,l[1]-sz),(l[0]+sz,l[1]+sz)],fill=(0,r,0))

    drawer.point(lt,fill=(0,128,0))
    drawer.rectangle([(new_box[0],new_box[1]),(new_box[2],new_box[3])])
    #pts = pts + 1.0



    return _out

def points_alone(past,pts,gt,k,path):
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.scatter(past[:,0],past[:,1],c='g')
    plt.scatter(pts[:, 0], pts[:, 1], c='r')
    plt.scatter(gt[:, 0], gt[:, 1], c='b')
    plt.savefig(path+"/tst"+str(k)+".png")
    return

def draw_simple_scenes(im, k, path, futures,past,gt,dim,weird,text=None,special=None):
    im=np.tile(im,[1,1,3])
    im=im*32

    im=im.astype('uint8')

    gf=Image.fromarray(im)

    gf=gf.resize((dim*4,dim*4))




    drawer = ImageDraw.Draw(gf)


    sz = 1

    poins = (np.array(past* 2, dtype=np.int32) )+ dim*2
    lt = [tuple(p) for p in poins.tolist()]
    for i, l in enumerate(lt):
        r = min(255, 160 + i * 10)
        drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(r, 0, 0))

    poins = (np.array(gt* 2, dtype=np.int32)) + dim*2
    lt = [tuple(p) for p in poins.tolist()]
    for i, l in enumerate(lt):
        r = min(255, 160 + i * 10)
        drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(0, 0, r))

    # poins = (np.array(weird * 4, dtype=np.int32)) + dim * 2
    # lt = [tuple(p) for p in poins.tolist()]
    # for i, l in enumerate(lt):
    #     r = min(255, 160 + i * 10)
    #     drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] + sz, l[1] + sz)], fill=(0, r, r))

    tr=0
    step=int(223/len(futures))
    for idx,future in enumerate(futures):

        poins = (np.array(future* 2, dtype=np.int32)) + dim*2
        lt = [tuple(p) for p in poins.tolist()]
        for i, l in enumerate(lt):

            r = min(255, 64 + tr * step)
            if (idx==special):
                drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(240, 255, 240))
            else:
                drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(0  , r, 0  ))
        tr+=1
    if(text):
        drawer.text((0,0),text,fill=(255,255,255))
    gf.save(path+"/im_t"+str(k)+".png")


def draw_scenes(sp_cord,sp_crop,im, k, path, futures,past,gt,dim,weird,text=None,special=None):
    im=np.tile(im,[1,1,3])
    im=im*32

    im=im.astype('uint8')

    gf=Image.fromarray(im)

    gf=gf.resize((dim*4,dim*4))

    sp_cord=(np.array(sp_cord* 2, dtype=np.int32) ) + dim*2

    h,w=sp_crop.shape[0:2]

    xx=np.tile(sp_crop[:,:,2:3],[1,1,3])

    xx=xx*50

    xx=xx.astype('uint8')


    im_cro=Image.fromarray(xx)
    im_cro=im_cro.resize((h *2, w*2 ))

    gf.paste(im_cro,(sp_cord[0]-im_cro.size[0]/2,sp_cord[1]-im_cro.size[1]/2))
    drawer = ImageDraw.Draw(gf)
    drawer.rectangle([(sp_cord[0] - 2, sp_cord[1] - 2), (sp_cord[0] + 2, sp_cord[1] + 2)], fill=(255, 255, 0))
    drawer.rectangle([(sp_cord[0] - 2-im_cro.size[0]/2, sp_cord[1] - 2-im_cro.size[0]/2), (sp_cord[0] + 2-im_cro.size[0]/2, sp_cord[1] + 2-im_cro.size[0]/2)], fill=(255, 255, 0))

    sz = 1

    poins = (np.array(past* 2, dtype=np.int32) )+ dim*2
    lt = [tuple(p) for p in poins.tolist()]
    for i, l in enumerate(lt):
        r = min(255, 160 + i * 10)
        drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(r, 0, 0))

    poins = (np.array(gt* 2, dtype=np.int32)) + dim*2
    lt = [tuple(p) for p in poins.tolist()]
    for i, l in enumerate(lt):
        r = min(255, 160 + i * 10)
        drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(0, 0, r))

    # poins = (np.array(weird * 4, dtype=np.int32)) + dim * 2
    # lt = [tuple(p) for p in poins.tolist()]
    # for i, l in enumerate(lt):
    #     r = min(255, 160 + i * 10)
    #     drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] + sz, l[1] + sz)], fill=(0, r, r))

    tr=0
    step=int(223/len(futures))
    for idx,future in enumerate(futures):

        poins = (np.array(future* 2, dtype=np.int32)) + dim*2
        lt = [tuple(p) for p in poins.tolist()]
        for i, l in enumerate(lt):

            r = min(255, 64 + tr * step)
            if (idx==special):
                drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(240, 255, 240))
            else:
                drawer.rectangle([(l[0] - sz, l[1] - sz), (l[0] , l[1] )], fill=(0  , r, 0  ))
        tr+=1
    if(text):
        drawer.text((0,0),text,fill=(255,255,255))
    gf.save(path+"/im_t"+str(k)+".png")

def draw_crops(imss,k,path,dps):
    ptss=[0,1,2,3,4,5,10,15,20,25,30,39]
    for p in ptss:
        ims=imss[p]
        tota=Image.new("L",(72*4,64))
        draw=ImageDraw.Draw(tota,mode="L")
        for ch in range(4):
            gf = Image.fromarray(ims[:,:,ch]*255.0)
            gf = gf.resize((64,64))
            tota.paste(gf,((8*ch)+(64*ch),0))
        draw.text((0,0),str(dps[p]),fill=(255))

        tota.save(path + "/crop_t" + str(k)+"_" +str(p)+ ".png")
