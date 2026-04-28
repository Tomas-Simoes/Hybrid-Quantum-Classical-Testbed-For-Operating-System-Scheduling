import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering

from data_contracts import (
    AffinityMatrix, Bundle, ClusteredSnapshot,
    DecompositorConfig, FeatureMatrix, ProcessInfo, SystemSnapshot,
)

class AdaptiveCluster:
    def __init__(self, decompositor_cfg: DecompositorConfig):
        self.dec_cfg = decompositor_cfg

    # entry point
    def decompose(self, snapshot: SystemSnapshot) -> ClusteredSnapshot:
        rt_procs, normal_procs = self._separate_rt_processes(snapshot)

        if not normal_procs:
            # only RT processes — trivial singleton decomposition
            result = self._trivial_decomposition(snapshot, rt_procs)
            result.rt_procs = rt_procs
            return result

        n = len(normal_procs)
        n_bundles = self.dec_cfg.num_bundles(n)

        if n_bundles >= n or n == 1:
            # No clustering needed — each normal process becomes its own bundle.
            # we pass normal_procs so RT processes are NOT re-included.
            result = self._trivial_decomposition(snapshot, normal_procs)
            result.rt_procs = rt_procs
            return result

        normal_snapshot = SystemSnapshot(
            timestamp=time.time(),
            snapshot_id=snapshot.snapshot_id,
            processes=normal_procs,
            num_cores=snapshot.num_cores,
            total_ram_mb=snapshot.total_ram_mb,
        )

        feature_matrix = self.build_feature_matrix(normal_snapshot)
        affinity_matrix = self.build_affinity_matrix(feature_matrix)

        try:
            clustered_snapshot = self.build_bundles(
                feature_matrix, affinity_matrix, normal_snapshot, n_bundles
            )
        except Exception as e:
            import warnings
            warnings.warn(f"SpectralClustering failed: ({e}). Using K-Means fallback.")
            clustered_snapshot = self._kmeans_fallback(
                feature_matrix, normal_snapshot, n_bundles
            )

        clustered_snapshot.num_cores = self.dec_cfg.num_cores
        clustered_snapshot.rt_procs = rt_procs
        return clustered_snapshot

    # w_eff helper: single definition used by every path
    def _compute_w_eff(self, proc: ProcessInfo) -> float:
        """
        Effective CPU demand, discounting time the process spent waiting on I/O.
        w_eff = cpu_weight * (1 - io_alpha * io_wait_ratio)
        Range: [0, cpu_weight].  io_wait_ratio is clamped to [0, 1] here.
        """
        io_ratio = max(0.0, min(1.0, proc.io_wait_ratio))
        return proc.cpu_weight * (1.0 - self.dec_cfg.io_alpha * io_ratio)

    # Feature / affinity construction
    def build_feature_matrix(self, snapshot: SystemSnapshot) -> FeatureMatrix:
        """
        Builds F where F[i, :] = [w_eff_i, rss_mb_i], then Z-score normalises.
        I/O wait is folded into w_eff rather than kept as a separate column.
        """
        raw_features = []
        pids = []
        w_effs = []

        for p in snapshot.processes:
            w_eff = self._compute_w_eff(p)
            raw_features.append([w_eff, p.rss_mb])
            pids.append(p.pid)
            w_effs.append(w_eff)

        F = np.array(raw_features)

        # Z-score normalisation per column
        means = np.mean(F, axis=0)
        stds = np.std(F, axis=0)
        stds[stds == 0] = 1.0  # avoid divide-by-zero when all values identical
        F_norm = (F - means) / stds

        return FeatureMatrix(
            F_norm=F_norm,
            pids=pids,
            F=F,
            w_eff=np.array(w_effs),
        )

    def build_affinity_matrix(self, feature_matrix: FeatureMatrix) -> AffinityMatrix:
        """
        Builds a combined RBF affinity matrix from CPU (w_eff) and memory (RSS).

        Kernel: A[i,j] = exp(-d^2 / (2 * sigma^2))
        Sigma is chosen adaptively as sqrt(median(d^2) / 2) so that the kernel
        evaluates to 1/e at the median squared distance.

        Final matrix: A = alpha * A_cpu + (1 - alpha) * A_mem
        Diagonal is zeroed before returning — SpectralClustering's normalised
        Laplacian is distorted by self-loops in the affinity matrix.
        """
        F = feature_matrix.F_norm  # columns: [0: w_eff, 1: RSS]
        alpha = self.dec_cfg.affinity_alpha

        cpu_col = F[:, 0].reshape(-1, 1)
        mem_col = F[:, 1].reshape(-1, 1)

        dist_cpu_sq = squareform(pdist(cpu_col, "sqeuclidean"))
        dist_mem_sq = squareform(pdist(mem_col, "sqeuclidean"))

        def adaptive_sigma(sq_dist_matrix: np.ndarray) -> float:
            upper = sq_dist_matrix[np.triu_indices_from(sq_dist_matrix, k=1)]
            median_sq_dist = np.median(upper)
            # sigma s.t. 2*sigma^2 = median_sq_dist  →  kernel = 1/e at median
            return float(np.sqrt(median_sq_dist / 2.0)) if median_sq_dist > 1e-9 else 1.0

        sigma_cpu = adaptive_sigma(dist_cpu_sq)
        sigma_mem = adaptive_sigma(dist_mem_sq)

        A_cpu = np.exp(-dist_cpu_sq / (2 * sigma_cpu ** 2))
        A_mem = np.exp(-dist_mem_sq / (2 * sigma_mem ** 2))

        A = alpha * A_cpu + (1 - alpha) * A_mem

        # Zero the diagonal: self-loops inflate node degree and distort the
        # normalised Laplacian used inside SpectralClustering.
        np.fill_diagonal(A, 0.0)

        return AffinityMatrix(A)

    # Bundle construction
    def build_bundles(
        self,
        feature_matrix: FeatureMatrix,
        affinity_matrix: AffinityMatrix,
        snapshot: SystemSnapshot,
        num_bundles: int,
    ) -> ClusteredSnapshot:
        """
        Runs SpectralClustering then applies quality-control checks:
          1. Intra-bundle heterogeneity via coefficient of variation (CV).
          2. Inter-bundle weight z-score outlier.
          3. Memory cap check.
        Bundles that fail any check (and have > 1 member) are recursively split.
        """
        F = feature_matrix.F
        A = affinity_matrix.A

        sc = SpectralClustering(n_clusters=num_bundles, affinity="precomputed", random_state=42)
        labels = sc.fit_predict(A)

        pid_to_proc = {p.pid: p for p in snapshot.processes}
        pid_to_weff = dict(zip(feature_matrix.pids, feature_matrix.w_eff))

        bundle_agg_cpu = np.zeros(num_bundles)
        bundle_agg_rss = np.zeros(num_bundles)
        bundle_pids: dict[int, list] = {i: [] for i in range(num_bundles)}

        for idx, label in enumerate(labels):
            pid = feature_matrix.pids[idx]
            bundle_pids[label].append(pid)
            bundle_agg_cpu[label] += pid_to_weff[pid]
            bundle_agg_rss[label] += pid_to_proc[pid].rss_mb

        bundle_weight_mean = np.mean(bundle_agg_cpu)
        bundle_weight_std = np.std(bundle_agg_cpu)
        if bundle_weight_std < 1e-9:
            bundle_weight_std = 1.0  # all bundles equal — no outliers possible

        W_total = sum(pid_to_weff.values())
        L_avg = W_total / snapshot.num_cores

        final_bundles = []
        next_bundle_id = 0

        for i in range(num_bundles):
            members = bundle_pids[i]
            if not members:
                continue

            member_weights = [pid_to_weff[p] for p in members]

            # 1. Intra-bundle heterogeneity via coefficient of variation.
            #    CV = (max - min) / mean — scale-invariant measure of spread.
            delta_w = max(member_weights) - min(member_weights)
            bundle_mean = np.mean(member_weights)
            if bundle_mean > 1e-9:
                cv = delta_w / bundle_mean
                is_heterogeneous = cv > self.dec_cfg.homogeneity_threshold
            else:
                is_heterogeneous = False  # all-zero weights: uniform, not heterogeneous

            # 2. Bundle weight z-score (this bundle vs all others)
            z_score = (bundle_agg_cpu[i] - bundle_weight_mean) / bundle_weight_std
            is_heavy_outlier = z_score > self.dec_cfg.zscore_threshold

            # 3. Memory pressure check
            mem_cap = snapshot.total_ram_mb / snapshot.num_cores
            is_mem_heavy = bundle_agg_rss[i] > (mem_cap * 0.4)

            if (is_heterogeneous or is_heavy_outlier or is_mem_heavy) and len(members) > 1:
                sub_bundles = self._split_bundle(
                    members, pid_to_weff, pid_to_proc, next_bundle_id, L_avg
                )
                final_bundles.extend(sub_bundles)
                next_bundle_id += len(sub_bundles)
            else:
                final_bundles.append(
                    Bundle(
                        bundle_id=next_bundle_id,
                        member_pids=members,
                        aggregate_cpu_weight=bundle_agg_cpu[i],
                        aggregate_rss_mb=bundle_agg_rss[i],
                        representative_cmd=f"Bundle_{next_bundle_id}",
                    )
                )
                next_bundle_id += 1

        return ClusteredSnapshot(
            bundles=final_bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id,
        )

    def _split_bundle(
        self, members, pid_to_weff, pid_to_proc, start_id, L_avg, depth=0
    ) -> list[Bundle]:
        MAX_DEPTH = 3
        weights = np.array([pid_to_weff[p] for p in members]).reshape(-1, 1)

        # Base case: singleton, max depth reached, or all weights identical
        if len(members) == 1 or depth >= MAX_DEPTH or np.allclose(weights, weights[0], atol=1e-5):
            # If identical weights but bundle is too heavy, force into singletons
            if len(members) > 1 and np.allclose(weights, weights[0], atol=1e-5):
                bundles = []
                for offset, pid in enumerate(members):
                    bundles.append(Bundle(
                        bundle_id=start_id + offset,
                        member_pids=[pid],
                        aggregate_cpu_weight=float(pid_to_weff[pid]),
                        aggregate_rss_mb=pid_to_proc[pid].rss_mb,
                        representative_cmd=f"Bundle_{start_id + offset}",
                    ))
                return bundles

            return [Bundle(
                bundle_id=start_id,
                member_pids=members,
                aggregate_cpu_weight=float(weights.sum()),
                aggregate_rss_mb=sum(pid_to_proc[p].rss_mb for p in members),
                representative_cmd=f"Bundle_{start_id}",
            )]

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        sub_labels = km.fit_predict(weights)

        result = []
        id_offset = 0

        for sub_id in range(2):
            sub_members = [members[j] for j, l in enumerate(sub_labels) if l == sub_id]
            if not sub_members:
                continue

            sub_agg = sum(pid_to_weff[p] for p in sub_members)

            if sub_agg > L_avg * 1.5 and len(sub_members) > 1:
                sub_result = self._split_bundle(
                    sub_members, pid_to_weff, pid_to_proc,
                    start_id + id_offset, L_avg, depth + 1,
                )
            else:
                sub_result = [Bundle(
                    bundle_id=start_id + id_offset,
                    member_pids=sub_members,
                    aggregate_cpu_weight=sub_agg,
                    aggregate_rss_mb=sum(pid_to_proc[p].rss_mb for p in sub_members),
                    representative_cmd=f"Bundle_{start_id + id_offset}",
                )]

            result.extend(sub_result)
            id_offset += len(sub_result)

        return result

    # Fallback / trivial paths
    def _kmeans_fallback(
        self, feature_matrix: FeatureMatrix, snapshot: SystemSnapshot, n_bundles: int
    ) -> ClusteredSnapshot:
        km = KMeans(n_clusters=n_bundles, random_state=42, n_init=10)
        labels = km.fit_predict(feature_matrix.F_norm)

        pid_to_proc = {p.pid: p for p in snapshot.processes}
        pid_to_weff = dict(zip(feature_matrix.pids, feature_matrix.w_eff))

        bundle_agg_cpu = np.zeros(n_bundles)
        bundle_agg_rss = np.zeros(n_bundles)
        bundle_pids: dict[int, list] = {i: [] for i in range(n_bundles)}

        for idx, label in enumerate(labels):
            pid = feature_matrix.pids[idx]
            bundle_pids[label].append(pid)
            bundle_agg_cpu[label] += pid_to_weff[pid]
            bundle_agg_rss[label] += pid_to_proc[pid].rss_mb

        bundles = [
            Bundle(
                bundle_id=i,
                member_pids=bundle_pids[i],
                aggregate_cpu_weight=bundle_agg_cpu[i],
                aggregate_rss_mb=bundle_agg_rss[i],
                representative_cmd=f"Bundle_{i}_fallback",
            )
            for i in range(n_bundles)
            if bundle_pids[i]
        ]

        return ClusteredSnapshot(
            bundles=bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id,
        )

    def _trivial_decomposition(
        self, snapshot: SystemSnapshot, procs: list | None = None
    ) -> ClusteredSnapshot:
        """
        Each process becomes its own singleton bundle.
        Uses _compute_w_eff for consistency with the clustering path.
        `procs` must be passed explicitly to avoid accidentally including RT processes.
        """
        target = procs if procs is not None else snapshot.processes
        bundles = [
            Bundle(
                bundle_id=i,
                member_pids=[p.pid],
                aggregate_cpu_weight=self._compute_w_eff(p),  # consistent with clustering path
                aggregate_rss_mb=p.rss_mb,
                representative_cmd=f"Bundle_{i}_singleton",
            )
            for i, p in enumerate(target)
        ]
        return ClusteredSnapshot(
            bundles=bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id,
        )

    def _separate_rt_processes(
        self, snapshot: SystemSnapshot
    ) -> tuple[list, list]:
        rt_procs = [p for p in snapshot.processes if p.priority_class == "RT"]
        be_procs = [p for p in snapshot.processes if p.priority_class != "RT"]
        return rt_procs, be_procs