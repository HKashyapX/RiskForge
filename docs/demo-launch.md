# Non-inferencing presentation service

RiskForge can expose its HTTP contract before the production model bundle is
installed. This mode is deliberately opt-in and cannot start when
`RISKFORGE_ENV` is `staging` or `production`.

## Launch

```sh
docker build -t riskforge-demo .
docker run --rm -p 8000:8000 \
  -e RISKFORGE_MODE=demo \
  -e RISKFORGE_ENV=development \
  riskforge-demo
```

Open `http://localhost:8000/docs` for the API contract.

- `GET /health` returns `200` because the HTTP process is live.
- `GET /ready` requires `X-Correlation-ID` and returns `503` because no model
  is installed.
- Inference endpoints return a sanitized `503 inference_unavailable` response.
- The service never generates substitute or fabricated safety predictions.

This mode is for presentation and API integration only. Production readiness
requires the ONNX model, its validated manifest, the matching input encoder,
and the concrete runtime dependency composer.
