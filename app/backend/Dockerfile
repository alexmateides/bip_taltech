FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --verbose -r requirements.txt

COPY . .

# FastAPI app is in /src/backend/__init__.py as `app`
CMD ["uvicorn", "src.__init__:app", "--host", "0.0.0.0", "--port", "8000"]
