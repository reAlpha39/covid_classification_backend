FROM python:3.9.17-alpine3.18

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5555

CMD ["python", "app.py"]

