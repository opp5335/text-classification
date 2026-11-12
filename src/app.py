# 1  导包
from flask import (  # flask 是一个 python 好用易上手的 web 框架
    Flask,  # 用来创建 APP
    request,  # 请求相关的
    jsonify  # python 结构 与 json str 之间的转换
)
from predictor import predict

# 2  创建 APP 服务
app = Flask(__name__)


# 3  创建一个 路径函数
# 装饰器：请求路径、请求方式
# @app.post("/predict")
@app.route("/predict", methods=['POST'])
def predict_api():  # 路径函数
    # 1  获取请求输入
    data = request.get_json()  # data = {“text”: xxx}

    # 2  校验: 验证获取的 data 是否满足预测要求
    # 400 错误 状态码
    if (not data) or ('text' not in data):
        return jsonify({'error': 'data is false'}), 400

    # 3  模型预测
    # result 是一个 dict
    result = predict(data)

    # 4  返回 json str
    return jsonify(result)


# 4  启动 APP
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',  # ip
        port=8000,  # 端口号
        debug=True,  # 调式模式，显示详细错误信息
    )
