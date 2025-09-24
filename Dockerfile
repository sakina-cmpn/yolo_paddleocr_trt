FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "benchmark.py"]
