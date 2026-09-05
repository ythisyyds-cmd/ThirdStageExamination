import gradio as gr
import requests
import mimetypes
import base64
from io import BytesIO
from PIL import Image

#根据选择请求对应接口 并读取返回的结果图片
def process_image(image_path, confidence, task_type):
    if image_path == None:
        raise gr.Error("请先上传图片")

    type_result = mimetypes.guess_type(image_path)    #根据扩展名获取文件类型
    content_type = type_result[0]

    if content_type == None:
        raise gr.Error("无法判断图片类型")

    if task_type == "目标检测":                      #根据处理类型选择后端接口
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

with gr.Blocks(title="Third Stage Examination") as demo:
    gr.Markdown("# 图片检测与分割")

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

    result_table = gr.Dataframe(
        headers=["类别", "置信度"],
        datatype=["str", "number"],
        interactive=False,                          #表格只用于显示识别结果
        label="识别结果"
    )

    process_button.click(                           #按顺序传入图片路径 置信度和处理类型
        fn=process_image,
        inputs=[input_image, confidence_threshold, task_choice],
        outputs=[output_image, result_table]
    )

demo.launch()
