import time
import psutil
from data_contracts import ProcessInfo, SystemSnapshot, TracerConfig


class ProcessTracer:
    def __init__(self, tracer_cfg: TracerConfig):
        self.min_rss = tracer_cfg.min_rss
        self.min_cpu = tracer_cfg.min_cpu
        self.cpu_interval = tracer_cfg.cpu_interval
        self.num_samples = tracer_cfg.num_samples

    def trace(self) -> SystemSnapshot:
        samples = {}
        rss_cache = {}
        info_cache = {}
        
        for _ in range(self.num_samples):
            for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_num", "nice", "memory_info"]):
                try:
                    mem = proc.info["memory_info"]
                    rss_mb = mem.rss / (1024 * 1024) if mem else 0.0

                    if rss_mb < self.min_rss:
                        continue

                    # first call to cpu_percent initializes measurement
                    proc.cpu_percent()
                    rss_cache[proc.info["pid"]] = rss_mb
                    info_cache[proc.info["pid"]] = proc.info
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            time.sleep(self.cpu_interval)

            for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_num", "nice", "memory_info"]):
                try:
                    pid = proc.info["pid"]
                    
                    if pid not in rss_cache:
                        continue
                    cpu = proc.cpu_percent() / 100.0
                    
                    if pid not in samples:
                        samples[pid] = []
                    
                    samples[pid].append(cpu)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        cpu_weights = {pid: max(vals) for pid, vals in samples.items()}
        cpu_avg_weights = {pid: sum(vals) / len(vals) for pid, vals in samples.items()}

        cpu_weights = cpu_avg_weights
        
        processes = []
        for pid, weight in cpu_weights.items():
            if weight < self.min_cpu:
                continue
            info = info_cache.get(pid)
            if not info:
                continue

            cmd = info["cmdline"]
            command = " ".join(cmd) if cmd else info["name"] or "unknown"

            processes.append(ProcessInfo(
                pid=pid,
                command=command,
                cpu_weight=weight,
                current_core=info["cpu_num"] or 0,
                rss_mb=rss_cache[pid],
                priority=info["nice"] or 0,
            ))

        return SystemSnapshot(
            timestamp=time.time(),
            num_cores=psutil.cpu_count(logical=True),
            processes=processes,
        )