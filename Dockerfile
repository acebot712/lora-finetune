FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y git ffmpeg libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/huggingface/diffusers

COPY . .

EXPOSE 7860

CMD ["python", "scripts/run_app.py"] 