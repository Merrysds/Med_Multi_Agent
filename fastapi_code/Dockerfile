# FROM python:3.10-slim
#pytorch里面自带纯python环境，不需要上面这句
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app

COPY requirements.txt .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
