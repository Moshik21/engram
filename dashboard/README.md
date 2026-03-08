# Engram Dashboard

The dashboard is the React 19 frontend for exploring the Engram graph in real time.
It talks to the FastAPI server over HTTP and `/ws/dashboard`, and in Docker it is
served by nginx with reverse proxies for `/api`, `/health`, and `/ws`.

## Development

```bash
cd dashboard
pnpm install
pnpm dev
```

The dev server runs on `http://localhost:5173` and proxies API/WebSocket traffic to
`http://localhost:8100` via [vite.config.ts](/Users/konnermoshier/Engram/dashboard/vite.config.ts).

## Commands

```bash
pnpm dev
pnpm test -- --run
pnpm lint
pnpm build
```

## Production

The Docker image builds the static app and serves it from nginx on port `8080`
inside the container. In the root compose stack it is published as `http://localhost:3000`.
