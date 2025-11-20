## Business proposal

> App that allows monitoring of emergency situations from public transport vehicle cameras

- The vehicle streams to server -> it gets analyzed for emergency situations (crashes…)
- When crash is identified a notification pops up, alerting the dispatcher
- The dispatcher analyzes the situation (camera recording, location, other metrics…) and possibly alerts the emergency services prompting a faster response.

## Running locally with Docker

1. Copy env files:
   - `cp .env.example .env` in repo root (fill backend secrets)
   - `cp src/frontend/.env.example src/frontend/.env` for frontend Vite vars
2. Start stack:
   ```sh
  docker-compose up --build
   ```
3. Backend available at `http://localhost:8000`, frontend Vite dev server at `http://localhost:5173` (uses `VITE_API_BASE_URL=http://backend:8000` over compose network).
4. Stop with `Ctrl+C` and `docker-compose down` when done.
