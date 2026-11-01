FROM python:3.10-slim

LABEL maintainer="DDoS Detection Project"
LABEL description="Interactive monitoring dashboard — LOCAL & MEC"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY measurements_seed.json .
COPY static ./static

RUN mkdir -p /app/data /app/logs
RUN cp measurements_seed.json /app/data/measurements.json || true

ENV DEPLOY_ENV=LOCAL
ENV HOST=0.0.0.0
ENV PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
