from flask import Flask, request, render_template
import predict
import os
import time

app = Flask(__name__)

# app.debug = True
app.config['UPLOAD_FOLDER'] = 'tmp'


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/index1')
def test():
    return app.send_static_file('index1.html')


@app.route('/predict', methods=['POST'])
def predictFromImg():
    if request.method == 'POST':
        # 实现异文件上传（开放式上传）
        predictImg = request.files['predictImg']
        # time.localtime():作用是格式化时间戳为本地的时间
        # time.mktime():返回用秒数来表示时间的浮点数．
        filename = str(int(time.mktime(time.localtime()))) + '.png'
        # os.path.join():将多个路径组合后返回
        predictImg.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imgurl = './tmp/' + filename
        result = predict.img2class(imgurl)
        print(result)

        return str(result)


if __name__ == '__main__':
    app.run(host='localhost', port=8888)
