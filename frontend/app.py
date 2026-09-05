import gradio as gr
import requests
import mimetypes
import base64
from io import BytesIO
from PIL import Image

#把图片发送给检测接口 并读取返回的结果图片
def process_image(image_path, confidence):
    if image_path == None:
        raise gr.Error("请先上传图片")

    type_result = mimetypes.guess_type(image_path)    #根据扩展名获取文件类型
    content_type = type_result[0]

    if content_type == None:
        raise gr.Error("无法判断图片类型")

    url = "http://127.0.0.1:8000/detect"
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

    return result_image

with gr.Blocks(title="Third Stage Examination") as demo:
    gr.Markdown("# 图片检测与分割")

    with gr.Row():
        input_image = gr.Image(
            type="filepath",                         #把上传图片的临时路径传给处理函数
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

    process_button.click(                           #按顺序把图片路径和置信度交给处理函数
        fn=process_image,
        inputs=[input_image, confidence_threshold],
        outputs=[output_image]
    )

demo.launch()