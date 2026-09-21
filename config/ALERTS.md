# What I would alert on (Task 3 monitoring notes)
#
# These are decisions, not code. Thresholds also live in config.yaml
# under monitoring. GET /metrics lists which alerts_triggered right now.

## Service health

1. error_rate_high
   - when rejected or failed requests / total >= alert_error_rate_high (default 5%)
   - meaning: bad payloads spiked, or the model path is broken

2. latency_high
   - when average latency_ms >= alert_latency_ms (default 2000 ms)
   - meaning: the service is too slow for an online check

## Prediction drift

3. late_rate_drift
   - when recent predicted late_rate >= alert_late_rate_high (default 15%)
   - and at least 20 scored rows (so one request does not panic us)
   - baseline_late_rate is about the test set late share (~6.6%)
   - meaning: input mix may have changed, or something is wrong upstream

## After real delivery dates arrive

4. Fill actual_is_late in logs/predictions.jsonl
   - then compute real precision / recall / f1_late offline
   - alert if f1_late drops a lot vs the notebook test value (0.2558)

## Where to look

- live numbers: GET /metrics
- request lines: logs/service.log
- scored rows for later eval: logs/predictions.jsonl
