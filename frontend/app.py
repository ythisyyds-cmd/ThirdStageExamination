import gradio as gr
import requests
import mimetypes
import base64
from io import BytesIO
from PIL import Image
import json
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException

#根据选择请求对应接口 并读取返回的结果图片
def process_image(image_path, confidence, task_type):
    if image_path == None:
        raise gr.Error("请先上传图片")

    type_result = mimetypes.guess_type(image_path)    #根据扩展名获取文件类型
    content_type = type_result[0]

    if content_type == None:
        raise gr.Error("无法判断图片类型")

    if task_type == "目标检测":                       #根据处理类型选择后端接口
        url = "http://127.0.0.1:8000/detect"
    elif task_type == "图像分割":
        url = "http://127.0.0.1:8000/segment"
    else:
        raise gr.Error("请选择处理类型")
    
    form_data = {"confidence": confidence}

    try:
        with open(image_path, "rb") as image_file:   #以二进制方式打开上传图片
            file_data = ("upload_image", image_file, content_type)
            files = {"image": file_data}

            response = requests.post(
                url,
                files=files,
                data=form_data,
                timeout=60
            )
    except requests.exceptions.RequestException:
        raise gr.Error("请求后端失败 请检查后端是否启动 或稍后重试")

    if response.status_code != 200:                         #请求失败时显示后端返回的信息
        raise gr.Error(response.text)

    response_data = response.json()                         #把返回的JSON解析成Python字典
    result_image_base64 = response_data["result_image"]

    result_image_bytes = base64.b64decode(
        result_image_base64
    )
    result_file = BytesIO(result_image_bytes)               #把还原的图片数据包装成文件
    result_image = Image.open(result_file)
    result_image.load()

    objects = response_data["objects"]                      #取出后端返回的目标列表
    table_data = []

    for item in objects:
        class_name = item["class_name"]
        object_confidence = item["confidence"]

        row = [class_name, object_confidence]               #每个目标对应表格的一行
        table_data.append(row)

    return result_image, table_data

#按通信方式处理图片 并逐次更新页面
def process_request(image_path, confidence, task_type, connection_type):
    if image_path == None:
        yield None, [], "请先上传图片"
        return

    if connection_type == "HTTP":
        yield None, [], "正在请求HTTP接口 等待处理结果"  #清除上次结果 并显示等待状态
        try:
            process_result = process_image(image_path, confidence, task_type)
        except gr.Error as error:
            yield None, [], str(error)
            return

        result_image = process_result[0]
        table_data = process_result[1]
        yield result_image, table_data, "处理完成"
        return

    if connection_type != "WebSocket":
        yield None, [], "请选择通信方式"
        return

    status_text = "正在连接后端"
    yield None, [], status_text                                 #yield先更新页面 函数随后继续执行

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    except OSError:
        yield None, [], "图片文件无法读取 请重新上传"
        return

    request_data = {"task_type": task_type, "confidence": confidence}
    request_text = json.dumps(request_data, ensure_ascii=False)
    url = "ws://127.0.0.1:8000/ws-process"

    try:
        with connect(url, max_size=20 * 1024 * 1024) as websocket:
            websocket.send(request_text)                        #先发参数 再发图片 与后端接收顺序一致
            websocket.send(image_bytes)

            while True:
                reply_text = websocket.recv(timeout=60)         #每条消息最多等待60秒
                reply_data = json.loads(reply_text)
                message_type = reply_data["type"]
                message = reply_data["message"]
                status_text = status_text + "\n" + message      #保留本次处理的状态记录

                if message_type == "status":
                    yield None, [], status_text
                elif message_type == "error":
                    yield None, [], status_text
                    return
                elif message_type == "result":
                    result_image_base64 = reply_data["result_image"]
                    result_image_bytes = base64.b64decode(result_image_base64)
                    result_file = BytesIO(result_image_bytes)
                    result_image = Image.open(result_file)
                    result_image.load()                         #还原后端发回的结果图片

                    objects = reply_data["objects"]
                    table_data = []
                    for item in objects:
                        class_name = item["class_name"]
                        object_confidence = item["confidence"]
                        row = [class_name, object_confidence]
                        table_data.append(row)

                    yield result_image, table_data, status_text
                    return                                      #收到最终结果后结束函数 并关闭连接
    except (OSError, WebSocketException, TimeoutError):
        status_text = status_text + "\n连接失败 中断或等待超时 请检查后端后重试"
        yield None, [], status_text


with gr.Blocks(title="Third Stage Examination") as demo:
    gr.Markdown("# 图片检测与分割")

    connection_choice = gr.Radio(
        choices=["HTTP", "WebSocket"],
        value="WebSocket",
        type="value",
        label="通信方式"
    )

    task_choice = gr.Radio(
        choices=["目标检测", "图像分割"],
        value="目标检测",
        type="value",                                #把选项文字传给函数
        label="处理类型"
    )

    with gr.Row():
        input_image = gr.Image(
            type="filepath",                         #把上传图片的临时路径传给函数
            sources=["upload"],
            label="上传图片"
        )

        output_image = gr.Image(
            label="结果图片",
            interactive=False
        )

    confidence_threshold = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.5,
        step=0.05,
        label="置信度阈值"
    )

    process_button = gr.Button(
        value="开始处理",
        variant="primary"
    )

    clear_button = gr.ClearButton(
        value="清空"                                                #先创建按钮 确定显示位置
    )

    status_output = gr.Textbox(
        label="处理状态",
        lines=4,
        interactive=False                                           #显示本次请求的状态和错误提示
    )

    result_table = gr.Dataframe(
        headers=["类别", "置信度"],
        datatype=["str", "number"],
        interactive=False,                                          #表格只用于显示识别结果
        label="识别结果"
    )

    clear_button.add(
        [input_image, output_image, result_table, status_output]    #同时清空结果和状态
    )

    process_button.click(                                           #输入和输出的顺序与函数对应
        fn=process_request,
        inputs=[input_image, confidence_threshold, task_choice, connection_choice],
        outputs=[output_image, result_table, status_output]
    )

demo.launch()
