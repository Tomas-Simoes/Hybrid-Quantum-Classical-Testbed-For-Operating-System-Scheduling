import time
import psutil
from contracts import ProcessInfo, SystemSnapshot


class ProcessTracer:
    def __init__(self, min_rss_mb: float = 10.0, min_cpu_weight: float = 10.0, cpu_interval: float = 1.0):
        self.min_rss_mb = min_rss_mb
        self.min_cpu_weight = min_cpu_weight
        self.cpu_interval = cpu_interval

    def trace(self, num_samples: int = 3) -> SystemSnapshot:
        samples = {}
        rss_cache = {}
        info_cache = {}
        
        for _ in range(num_samples):
            for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_num", "nice", "memory_info"]):
                try:
                    mem = proc.info["memory_info"]
                    rss_mb = mem.rss / (1024 * 1024) if mem else 0.0

                    if rss_mb < self.min_rss_mb:
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

        processes = []
        for pid, weight in cpu_weights.items():
            if weight < self.min_cpu_weight:
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