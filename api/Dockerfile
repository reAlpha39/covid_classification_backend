FROM python:3.9.17-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5555

CMD ["python", "app.py"]

