FROM agrigorev/model-2025-hairstyle:v1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]