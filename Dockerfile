FROM python:3.12-slim
RUN groupadd -r app && useradd -r -g app app
WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./

COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh

RUN apt-get update && apt-get install -y curl unzip awscli

RUN pip install uv \
    && uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]