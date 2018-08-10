import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import numpy as np


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()#使用BytesIO操作二进制数据
    return PIL.Image.fromarray(a) #.save(f, fmt)
    #display(Image(data=f.getvalue()))#获取写入的数据


def showtensor(a):
    #参数a是numpy类型
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[ :, :, :]
   # inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    return showarray(inp)
    #clear_output(wait=True)#Clear the output of the current cell receiving output.
