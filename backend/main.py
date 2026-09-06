from fastapi import WebSocket
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
if __name__ == "__main__":
    from yolo_func import detect_image, segment_image
else:
    from backend.yolo_func import detect_image, segment_image
from io import BytesIO
from PIL import Image

app = FastAPI()                                      #创建FastAPI应用

#检查上传文件的大小和内容类型
def check_image(image, image_bytes):
    image_size = len(image_bytes)                    #获取上传内容的字节数

    if image_size == 0:
        raise HTTPException(
            status_code=400,
            detail="上传的图片内容为空"
        )

    content_type = image.content_type                #获取请求中标明的文件类型

    if content_type == None:
        raise HTTPException(
            status_code=400,
            detail="没有提供文件类型"
        )

    is_image = content_type.startswith("image/")     #判断类型是否以image/开头

    if is_image == False:
        raise HTTPException(
            status_code=400,
            detail="请上传图片文件"
        )
    try:
        image_file = BytesIO(image_bytes)            #把上传内容包装成可以读取的文件
        checked_image = Image.open(image_file)
        checked_image.load()                         #实际读取图片内容
        checked_image.close()
    except OSError:
        raise HTTPException(
            status_code=400,
            detail="图片无法读取，请检查文件是否损坏"
        )

    

@app.get("/")                                        #收到访问根地址的GET请求时执行下面的函数
def read_root():
    return {"message": "API is running"}             #字典会自动转换成JSON返回


@app.post("/upload-test")                            #接收网页上传的图片和置信度
async def upload_test(
    image: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    image_bytes = await image.read()                 #读取上传图片的二进制数据

    return {
        "filename": image.filename,
        "confidence": confidence,
        "image_size": len(image_bytes)
    }


@app.post("/detect")                                 #接收图片并返回YOLO目标检测结果
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(0.5, ge=0.1, le=1.0)    #限制置信度范围
):
    image_bytes = await image.read()
    check_image(image, image_bytes)                  #处理前检查上传文件
    detection_result = detect_image(
        image_bytes,
        confidence
    )

    detection_data = detection_result[0]             #分别取出检测数据和结果图片
    result_image_base64 = detection_result[1]

    return {
        "filename": image.filename,
        "confidence": confidence,
        "objects": detection_data,
        "result_image": result_image_base64
    }


@app.post("/segment")                                #接收图片并返回YOLO图像分割结果
async def segment(
    image: UploadFile = File(...),
    confidence: float = Form(0.5, ge=0.1, le=1.0)
):
    image_bytes = await image.read()
    check_image(image, image_bytes)                  #处理前检查上传文件
    segmentation_result = segment_image(
        image_bytes,
        confidence
    )

    segmentation_data = segmentation_result[0]       #分别取出分割数据和结果图片
    result_image_base64 = segmentation_result[1]

    return {
        "filename": image.filename,
        "confidence": confidence,
        "objects": segmentation_data,
        "result_image": result_image_base64
    }


@app.websocket("/ws-test")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()                          #接受客户端连接

    request_data = await websocket.receive_json()     #先接收处理参数 自动解析成字典
    task_type = request_data["task_type"]
    confidence = request_data["confidence"]

    image_bytes = await websocket.receive_bytes()     #再接收图片的二进制数据
    image_size = len(image_bytes)

    if image_size == 0:
        error_data = {"message": "上传的图片内容为空"}
        await websocket.send_json(error_data)         #把错误信息发回客户端
        await websocket.close()
        return                                        #结束函数

    try:
        image_file = BytesIO(image_bytes)             #把收到的图片数据包装成文件
        checked_image = Image.open(image_file)
        checked_image.load()                          #实际读取图片 检查内容是否有效
        checked_image.close()
    except OSError:
        error_data = {"message": "图片无法读取 请检查文件内容"}
        await websocket.send_json(error_data)
        await websocket.close()
        return

    reply = {
        "message": "图片数据已收到",
        "task_type": task_type,
        "confidence": confidence,
        "image_size": image_size
    }
    await websocket.send_json(reply)                 #把接收情况转换成JSON发回客户端
    await websocket.close()


if __name__ == "__main__":                          
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
