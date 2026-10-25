# Device Recall Explainer. Defaults to the offline sample replay, so it runs with no keys and no services.
# Live run: docker run --rm --env-file .env <image> "<question>" --playbook   (set SAMPLE_MODE=false)
FROM node:22-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY engines/hmem/reference/package.json engines/hmem/reference/package-lock.json engines/hmem/reference/
RUN cd engines/hmem/reference && npm ci --ignore-scripts

COPY . .

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONWARNINGS=ignore \
    SAMPLE_MODE=true

ENTRYPOINT ["python", "-m", "agent.main"]
CMD ["Why did cardiovascular device recall records rise between 2023 and 2024, and which product codes drove it?", "--playbook", "--memory-dir", "/tmp/hmem", "--no-log"]
