# tracer

This module captures live system process data.

## ProcessTracer

`process_tracer.py` uses `psutil` to build a `SystemSnapshot`.

It records:

- PID,
- command,
- current CPU core,
- CPU weight,
- RSS memory in MB,
- priority/nice value,
- estimated I/O wait ratio,
- priority class (`RT` or `BE` when available).

The tracer samples CPU activity over `TracerConfig.cpu_interval`.

CPU weights are normalised to total logical CPU capacity:

```text
cpu_weight = (psutil_process_cpu_percent / 100) / logical_core_count
```

Values are capped to `[0, 1]` to protect downstream QUBO heuristics from
sampling overshoot. For example, a process using one full core on an 8-core
machine has `cpu_weight ~= 0.125`.

## TracerConfig

Important fields:

- `min_rss`: minimum resident memory threshold.
- `min_cpu`: CPU threshold, currently not enforced in the main trace loop.
- `cpu_interval`: sampling interval.
- `num_samples`: reserved/configured sample count.
- `live_mode`: UI/engine flag deciding live tracing vs preset snapshot.

## Notes

`tracer.py` is an older stub and is not the main runtime path.
