from websockets.sync.client import connect
import json
import base64
from io import BytesIO
from PIL import Image

url = "ws://127.0.0.1:8000/ws-process"
image_path = input("请输入图片路径：")

with open(image_path, "rb") as image_file:                       #以二进制方式读取本地图片
    image_bytes = image_file.read()

request_data = {
    "task_type": "目标检测",
    "confidence": 0.5
}
request_text = json.dumps(request_data, ensure_ascii=False)      #把字典转换成JSON文字

with connect(url, max_size=20 * 1024 * 1024) as websocket:       #允许接收最大20MB的结果消息
    websocket.send(request_text)                                 #先发送处理参数
    websocket.send(image_bytes)                                  #再发送图片数据

    while True:
        reply_text = websocket.recv()                            #持续接收后端分次发送的消息
        reply_data = json.loads(reply_text)
        message_type = reply_data["type"]                        #根据消息类型决定下一步
        print(reply_data["message"])

        if message_type == "error":
            break                                                #出错后结束接收

        if message_type == "result":
            objects = reply_data["objects"]
            print("识别结果：", objects)

            result_image_base64 = reply_data["result_image"]
            result_image_bytes = base64.b64decode(result_image_base64)
            result_file = BytesIO(result_image_bytes)            #把结果图片还原后打开
            result_image = Image.open(result_file)
            result_image.load()
            result_image.show()
            break                                                #收到最终结果后结束接收
