FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app_pro.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
