from io import BytesIO
from PIL import Image
from ultralytics import YOLO


model = YOLO("yolov8n.pt")                           #启动后端时加载一次模型


#接收图片的二进制数据并完成目标检测
def detect_image(image_bytes, confidence_threshold):
    image_file = BytesIO(image_bytes)                 #把二进制数据转换成类似文件的形式
    image = Image.open(image_file)
    image = image.convert("RGB")

    results = model(image, conf=confidence_threshold)
    result = results[0]

    detection_data = []

    for box in result.boxes:
        class_id = int(box.cls.item())                #取出类别id和置信度
        confidence = box.conf.item()
        class_name = result.names[class_id]

        coordinates = box.xyxy[0].tolist()            #取出检测框左上角和右下角坐标
        rounded_coordinates = []

        for coordinate in coordinates:
            rounded_coordinates.append(round(coordinate, 1))

        detection_data.append({
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "box": rounded_coordinates
        })

    return detection_data