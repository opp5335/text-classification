# 1  导包
import requests  # 用来发送请求
import time  # 记录时间

# 2  定义 预测接口地址
url = 'http://127.0.0.1:8000/predict'

# 3  构造请求数据 data
data = {"text": "中国人民公安大学2012年硕士研究生目录及书目"}

# 4  记录开始时间
start_time = time.time()

# 5  发送请求 try，得到响应
try:
    response = requests.post(  # response 服务端返回的响应
        url,  # 发送给谁
        json=data,  # 发送的内容
    )

    # 6  计算耗时，单位是 ms 毫秒
    total_time = (time.time() - start_time) * 1000

    # 7  检查响应状态 200 OK
    if response.status_code == 200:
        result = response.json()
        print(f"预测结果为: {result['pred_class']}")
    else:
        print(f"请求失败: {response.status_code}")

# 8  try 失败，执行 except
except Exception as e:
    print(f"请求出错: {str(e)}")
