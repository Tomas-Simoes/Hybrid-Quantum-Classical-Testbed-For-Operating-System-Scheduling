from tabulate import tabulate

from data_contracts import ClusteredSnapshot, SystemSnapshot

class SnapshotVisualizer:
    @staticmethod
    def print_system_snapshot(snap: SystemSnapshot, title="RAW SYSTEM SNAPSHOT", limit=15):
        print(f"\n{'='*80}")
        print(f" {title} ")
        print(f" ID: {snap.snapshot_id} | Cores: {snap.num_cores} | Time: {snap.timestamp}")
        print(f"{'='*80}")
        
        header = f"{'PID':>8} | {'CLASS':>4} | {'CPU%':>6} | {'MEM(MB)':>9} | {'I/O':>5} | {'COMMAND'}"
        print(header)
        print("-" * len(header))
        
        # Sort by CPU to see what's heavy
        sorted_procs = sorted(snap.processes, key=lambda x: x.cpu_weight, reverse=True)
        
        for p in sorted_procs[:limit]:
            cmd = (p.command[:50] + '..') if len(p.command) > 50 else p.command
            print(f"{p.pid:8} | {p.priority_class:4} | {p.cpu_weight*100:5.1f}% | {p.rss_mb:8.1f} | {p.io_wait_ratio:5.2f} | {cmd}")
        
        if len(snap.processes) > limit:
            print(f"... and {len(snap.processes) - limit} more processes.")

    @staticmethod
    def print_clustered_snapshot(c_snap: ClusteredSnapshot):
        print(f"\n{'='*80}")
        print(f" ADAPTIVE CLUSTERING RESULT ")
        print(f" Source ID: {c_snap.source_snapshot_id} | Bundles: {len(c_snap.bundles)}")
        print(f"{'='*80}")
        
        header = f"{'B_ID':>5} | {'AGG CPU%':>10} | {'AGG MEM(MB)':>12} | {'MEMBERS (PIDs)'}"
        print(header)
        print("-" * len(header))
        
        for b in sorted(c_snap.bundles, key=lambda x: x.aggregate_cpu_weight, reverse=True):
            pids = ", ".join(map(str, b.member_pids))
            print(f"{b.bundle_id:5} | {b.aggregate_cpu_weight*100:9.1f}% | {b.aggregate_rss_mb:11.1f} | [{pids}]")
        print(f"{'='*80}\n")