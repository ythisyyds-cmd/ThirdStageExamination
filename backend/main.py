from fastapi import FastAPI, UploadFile, File, Form
from backend.yolo_func import detect_image

app = FastAPI()                                      #创建FastAPI应用


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
    confidence: float = Form(0.5)
):
    image_bytes = await image.read()
    detection_data = detect_image(
        image_bytes,
        confidence
    )

    return {
        "filename": image.filename,
        "confidence": confidence,
        "objects": detection_data
    }