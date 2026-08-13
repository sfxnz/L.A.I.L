# --- API (Bun controller) ---
FROM oven/bun:1.2 AS api
WORKDIR /app
COPY package.json bun.lock bunfig.toml ./
COPY packages/shared packages/shared
COPY packages/backend packages/backend
RUN bun install --frozen-lockfile
ENV LAIL_HOST=0.0.0.0
ENV LAIL_INSECURE_BIND=1
ENV LAIL_API_PORT=8787
EXPOSE 8787
CMD ["bun", "run", "packages/backend/src/index.ts"]

# --- Serve engine (Python) ---
# Console API only — no Docker CLI, no GPU. Start vLLM on the host.
FROM python:3.12-slim AS serve-engine
WORKDIR /app
COPY packages/serve-engine/pyproject.toml packages/serve-engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/serve-engine /app
ENV PYTHONPATH=/app
ENV LAB_API_PORT=8765
ENV LAB_HOST=0.0.0.0
ENV LAIL_HOST=0.0.0.0
ENV LAIL_INSECURE_BIND=1
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]

# --- Web (Next.js) ---
FROM oven/bun:1.2 AS web-build
WORKDIR /app
COPY package.json bun.lock bunfig.toml ./
COPY packages/shared packages/shared
COPY apps/web apps/web
RUN bun install --frozen-lockfile
WORKDIR /app/apps/web
RUN bun run build

FROM oven/bun:1.2 AS web
WORKDIR /app/apps/web
COPY --from=web-build /app /app
ENV PORT=3000
EXPOSE 3000
CMD ["bun", "run", "start"]
