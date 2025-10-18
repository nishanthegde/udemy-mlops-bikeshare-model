FROM python:3.13-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
CMD ["gunicorn", "-b", ":8080", "app:app"]