FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

COPY ../requirements.txt .
RUN pip install -r requirements.txt

COPY ../Code/ .

CMD ["python3", "main.py", "--demo"]