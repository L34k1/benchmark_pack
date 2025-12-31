# Web Security Scan (Passive) MVP

Minimal MVP for a passive web security scan pipeline using OWASP ZAP Baseline + lightweight probes.

## Prerequisites

- Docker + Docker Compose
- Authorized target URL (HTTP/HTTPS)

## Quick start

```bash
docker compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:5173

## Example run

1. Open the frontend UI.
2. Enter a target URL such as `https://example.com`.
3. Click **Lancer le scan**.
4. Monitor status and download the HTML report.

Artifacts for a run are stored under:

```
/data/runs/{run_id}/merged_findings.json
/data/runs/{run_id}/report.html
/data/runs/{run_id}/zap/
```

## API

- `POST /runs` Body: `{ "target_url": "https://example.com", "max_duration_sec": 300 }`
- `GET /runs/{run_id}` Status and progress
- `GET /runs/{run_id}/findings` Normalized findings
- `GET /runs/{run_id}/report.html` Downloadable HTML report

Status values: `queued | running | finished | failed`

## Security and limitations

- Passive scans only, no destructive tests
- ZAP Baseline uses passive analysis
- Timeouts and max duration enforced per run
- Light probes only (TLS and header/cookie checks)

## Legal disclaimer

Use this tool **only** against targets you own or have explicit permission to test. This project is intended for authorized security assessments and training. Unauthorized scanning may be illegal.
