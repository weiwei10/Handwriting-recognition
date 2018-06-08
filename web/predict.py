import matplotlib.image as mpimg
import numpy as np
# from PIL import Image
from keras.models import model_from_json
# import h5py
import tensorflow as tf

graph = tf.get_default_graph()
# 一种轻量级的数据交换格式
model = model_from_json(open('./model/my_model_architecture.json').read())

model.load_weights('./model/my_model_weights.h5')
# f = h5py.File("./model/model.h5")
# model.load_weights('./model/model.h5')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # dot()返回的是两个数组的点积


def img2class(imgFile):
    img = mpimg.imread(imgFile)  # 读取图片.
    print(img.shape)
    img = rgb2gray(img)  # 将真彩色图像RGB转换为灰度强度图像.
    print(img)
    print(img.shape)
    img = img.reshape(1, 784)
    # img = img.reshape(1, 3136)
    global graph  # “投放”到session中的图
    with graph.as_default():
        pre = model.predict_classes(img)
        # pre = f.predict_classes(img)
        result = pre[0]
    return result
    # return pre


# def img2class(imgFile):
#     # Image.open()打开文件，返回一个image对象;打开后，我们可以查看一些图片信息，如im.format, im.size, im.mode
#     # convert()转换图片格式
#     img = Image.open(imgFile).convert('1')
#     if img.size[0] != 28 or img.size[1] != 28:
#             img = img.resize((28, 28))
#     imgNew = []
#     for i in range(28):
#         for j in range(28):
#             pixel = 1.0 - float(img.getpixel((j, i)))/255.0
#             pixel = 1.0-pixel
#             imgNew.append(pixel)
#     imgNew = np.array(imgNew)
#     imgNew = imgNew.reshape(1, 784)

#     pre = model.predict_classes(imgNew)
#     result = pre[0]
#     return result
