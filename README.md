<img src="/assets/logo.svg" alt="Description" width="auto" height="100">

## Business proposal

> App that reports emergency situations from (public transport) vehicle dashcams, allowing for faster emergency services response.

- The dashcam stream gets analyzed (either in the vehicle or at central server)
- When an emergency event is identified, an operator is alerted, analyzes the situation and optionally dispatches the emergency services
- Location metadata speeds up the process
- The faster response can save someone's life

## Running locally with Docker

1. Copy env files:
   - `cp .env.example .env` in repo root (fill backend secrets)
   - `cp src/frontend/.env.example src/frontend/.env` for frontend Vite vars
2. Start stack:
   
   ```sh
   docker-compose up --build
   ```
3. Frontend (Web App) available at `http://localhost:5173`.
4. Backend available at `http://localhost:8000`.
5. Stop with `Ctrl+C` or `docker-compose down` when done.
