from websockets.sync.client import connect
import json

url = "ws://127.0.0.1:8000/ws-test"
image_path = input("请输入图片路径：")

with open(image_path, "rb") as image_file:           #以二进制方式读取本地图片
    image_bytes = image_file.read()

request_data = {
    "task_type": "目标检测",
    "confidence": 0.5
}
request_text = json.dumps(request_data, ensure_ascii=False)  #把字典转换成JSON文字

with connect(url) as websocket:
    websocket.send(request_text)                   #先发送处理参数
    websocket.send(image_bytes)                    #再发送图片数据

    reply_text = websocket.recv()                  #读取后端返回的JSON文字
    reply_data = json.loads(reply_text)            #把JSON文字转换回字典
    print(reply_data)