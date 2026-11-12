# 1  导包
import streamlit as st
import requests  # 发送请求
import time  # 计算耗时

# todo 命令行执行
#  streamlit run streamlit_app.py


# 2  创建 页面标题
st.title("文本分类预测")

# 3  创建 输入文本框
text_input = st.text_area(
    "请你输入要预测的文本:",  # 给用户的提示词
    "中国人民公安大学2012年硕士研究生目录及书目",  # 文本框的默认值
)

# 4  创建 预测按钮
if st.button("预测"):
    # 5  构建 请求数据 和 请求地址url
    data = {'text': text_input}
    url = 'http://127.0.0.1:8000/predict'

    # 6  记录开始时间
    start_time = time.time()

    # 7  发送请求 try，得到响应
    try:
        response = requests.post(
            url,  # 发送给谁
            json=data,  # 发送的内容
        )
        # 8  计算耗时 ms
        total_time = (time.time() - start_time) * 1000
        # 9  检查响应状态码 200
        if response.status_code == 200:
            result = response.json()  # dict {'text':xxx, 'pred_class':xxx}
            label = result['pred_class']
            st.success(f"预测分类: {label}")
            st.info(f"请求耗时: {total_time:.2f} ms")
        else:
            st.error(f"请求失败: {response.status_code}")
    except Exception as e:
        st.error(f"请求出错: {str(e)}")

# 10  运行提示
st.write("请确保 Flask API 服务已经在 localhost:8000 运行")
