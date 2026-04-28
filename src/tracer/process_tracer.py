import time
import psutil
import os
from data_contracts import ProcessInfo, SystemSnapshot, TracerConfig
import uuid

class ProcessTracer:
    def __init__(self, tracer_cfg: TracerConfig):
        self.min_rss = tracer_cfg.min_rss
        self.min_cpu = tracer_cfg.min_cpu
        self.cpu_interval = tracer_cfg.cpu_interval
        self.num_samples = tracer_cfg.num_samples

    def trace(self) -> SystemSnapshot:
        initial_stats = {}
        procs_cache: dict[int, psutil.Process] = {}
        
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_num", "nice", "memory_info", "cpu_times"]):
            try:
                pid = proc.info["pid"] 
                mem = proc.info["memory_info"]
                rss_mb = mem.rss / (1024 * 1024) if mem else 0.0

                if rss_mb < self.min_rss:
                    continue

                # initializes measurement
                proc.cpu_percent()
                procs_cache[pid] = proc
                
                initial_stats[pid] = {
                    "times": proc.cpu_times(),
                    "info": proc.info,
                    "rss": rss_mb
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        time.sleep(self.cpu_interval)
        final_proc = []
        
        for pid, proc in procs_cache.items():
            try:
                pid = proc.info["pid"]
                
                if pid not in initial_stats:
                    continue
                
                # Calculate delta for I/O wait
                now_times = proc.cpu_times()
                prev_times = initial_stats[pid]["times"]
                
                io_now = getattr(now_times, 'iowait', 0.0)
                io_prev = getattr(prev_times, 'iowait', 0.0)
                io_delta = io_now - io_prev
                
                total_delta = sum(now_times)- sum(prev_times)

                weight_i = proc.cpu_percent() / 100.0
                mem_i = initial_stats[pid]["rss"]
                ioratio_i = (io_delta / total_delta) if total_delta > 0 else 0.0

                # if weight_i < self.min_cpu: continue TODO CHECK IF WE NEED THIS

                info = initial_stats[pid]["info"]
                final_proc.append(ProcessInfo(
                    pid=pid,
                    cpu_weight=weight_i,
                    rss_mb=mem_i,
                    io_wait_ratio=ioratio_i,
                    command=" ".join(info["cmdline"]) if info["cmdline"] else info["name"],
                    current_core=info["cpu_num"] or 0,
                    priority=info["nice"] or 0,
                    priority_class= self.get_priority_class(pid)
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
        
        return SystemSnapshot(
            timestamp=time.time(),
            num_cores=psutil.cpu_count(logical=True),
            total_ram_mb = psutil.virtual_memory().total / (1024 * 1024),
            processes=final_proc,
            snapshot_id=str(uuid.uuid4()) 
        )
    
    def get_priority_class(self, pid: int) -> str: 
        try:
            policy = os.sched_getscheduler(pid)

            if policy in (os.SCHED_FIFO, os.SCHED_RR):
                return "RT"
            else:
                return "BE"
        except (ProcessLookupError, PermissionError):
            import warnings
            warnings.warn(f"Priority class for PID {pid} defaulting to BE")
            return "BE" # default to BE if we can't inspect it