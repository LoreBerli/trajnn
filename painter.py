import numpy as np
from tkinter import *
import PIL
import copy
from PIL import Image, ImageDraw,ImageTk
import os
import shutil
table=[
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216), (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230), (191, 239, 69),(250, 190, 190), (70, 153, 144), (230, 190, 255), (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 216, 177), (0, 0, 117), (169, 169, 169), (255, 255, 255), (0, 0, 0)
]




def save():
    global image_number
    filename = f'image_{image_number}.png'   # image_number increments by 1 at every save
    dr = str(vid)+"_"+str(frame) + "_MOD_CONTEXT_TEST"
    #image1.save(filename)
    tobesaved=image1.convert(mode="L")
    tobesaved.save(dr+"/"+filename)
    ennepi=np.array(tobesaved)
    np.savez(dr+"/modified.npz",seg_map=ennepi)

    image_number += 1


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    cv.bind('<B2-Motion>', pick)
    lastx, lasty = e.x, e.y

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

def paint(e):
    global lastx, lasty,color
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=10,fill=_from_rgb(table[color]))
    cv.create_oval(x-15,y-15,x+15,y+15,fill=_from_rgb(table[color]),outline=_from_rgb(table[color]))
    #  --- PIL
    #draw.line((lastx, lasty, x, y), fill=_from_rgb(table[1]), width=10)
    draw.ellipse([x-15,y-15,x+15,y+15], fill=color, width=1)
    lastx, lasty = x, y

def transform(im):
    pix=im.load()
    other=Image.new("RGB",im.size)
    otherp=other.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            vg=im.getpixel((x,y))
            otherp[x, y] = table[vg]

    return other

def recover_image_from_segmentation(vid,frame):
    npseg=np.load("/home/cioni/data/kitti_rev2/training/"+str(vid)+"/deeplab_cache/"+str(frame).zfill(8)+".npz")
    segm=npseg.f.seg_map
    to_im=Image.fromarray(segm,mode="I")
    return copy.deepcopy(to_im),segm
def pick(e):
    global lasty,lastx,color
    x, y = e.x, e.y
    print(lastx)
    print(lasty)
    vg = bak.getpixel((x,y))
    lastx, lasty = x, y
    color=vg


vid=12
frame=12



root = Tk()

lastx, lasty = None, None
color=1
image_number = 0
bak,orig=recover_image_from_segmentation(vid,frame)
dr = str(vid)+"_"+str(frame) + "_MOD_CONTEXT_TEST"
os.mkdir(dr)
np.savez(dr+"/original.npz",seg_map=orig)

back=transform(bak)
cv = Canvas(root, width=back.width, height=back.height, bg='white')

bkk=image = ImageTk.PhotoImage(back)
cv.create_image(back.width/2,back.height/2,image=bkk)
# --- PIL

image1,_ = recover_image_from_segmentation(vid,frame)#PIL.Image.new('RGB', (640, 480), 'white')
image1.show()
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)

cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

root.mainloop()