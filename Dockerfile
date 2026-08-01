# Image officielle Playwright : Chromium + toutes les libs système déjà installées.
# Le tag doit matcher la version de playwright dans requirements.txt.
FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers=2 --threads=4 --timeout=120
