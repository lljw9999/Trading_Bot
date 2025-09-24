# Contributing Guide

Thanks for improving the Multi-Layer Trading System!

## Local Test Strategy

We stage tests the same way as CI. Install dependencies and run:

```bash
# 1. Unit suite (fast)
pip install -r requirements.txt
pytest -m "not integration and not soak" --maxfail=5 --durations=20

# 2. Integration suite (Redis/API mocks)
docker compose -f docker-compose.ci.yml up -d
pytest -m "integration and not soak" --maxfail=5 --durations=20
# remember to stop services
docker compose -f docker-compose.ci.yml down

# 3. Soak/nightly tests (optional)
pip install .[ml,onnx,bandits]
pytest -m "soak" --maxfail=5 --durations=20
```

## Environment Flags

| Variable       | Default | Purpose                                            |
|----------------|---------|----------------------------------------------------|
| `DRY_RUN`      | `1`     | Prevent live orders. Set to `0` explicitly for prod |
| `OPENAI_MOCK`  | `1`     | Skip OpenAI calls during tests/CI                   |
| `REDIS_MOCK`   | `0`     | Use real Redis unless explicitly mocked             |
| `USE_NOWNODES` | `0`     | Enable NOWNodes websocket connector + tests         |

## Pull Request Checklist

- [ ] Unit suite passes locally (`pytest -m "not integration and not soak"`)
- [ ] Added/updated tests for new behaviour
- [ ] Runbook / docs updated if behaviour changes or new flags introduced
- [ ] CHANGELOG entry for user-visible changes
- [ ] CI workflow passes (unit + integration)

Happy shipping!
