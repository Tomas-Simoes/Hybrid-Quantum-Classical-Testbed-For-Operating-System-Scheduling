import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

from data_contracts import AffinityMatrix, Bundle, ClusteredSnapshot, DecompositorConfig, FeatureMatrix, SystemSnapshot

class Decompositor:
    def __init__(self, decompositor_cfg: DecompositorConfig):
        self.dec_cfg = decompositor_cfg

    def decompose(self, snapshot: SystemSnapshot):
        feature_matrix = self.build_feature_matrix(snapshot)
        affinity_matrix = self.build_affinity_matrix(feature_matrix)
        clustered_snapshot = self.build_bundles(feature_matrix, affinity_matrix, snapshot)

        return clustered_snapshot

    def build_feature_matrix(self, snapshot: SystemSnapshot) -> FeatureMatrix:
        raw_features = []
        pids = []

        for p in snapshot.processes:
            # I/O correction to CPU weight
            w_eff = p.cpu_weight * (1 - self.dec_cfg.io_alpha * p.io_wait_ratio)

            raw_features.append([w_eff, p.rss_mb])
            pids.append(p.pid)
        
        F = np.array(raw_features)

        # Z-score normalization
        means = np.mean(F, axis=0)
        stds = np.std(F, axis=0)

        stds[stds == 0] = 1.0

        F_norm = (F - means) / stds
        return FeatureMatrix(
            F_norm=F_norm,
            pids=pids,
            F=F,
        )

    def build_affinity_matrix(self, feature_matrix: FeatureMatrix) -> AffinityMatrix:
        # F_norm has columns: [0: CPU, 1: RSS] (IO ratio is embeded in CPU already) 
        F = feature_matrix.F_norm
        alpha = self.dec_cfg.affinity_alpha
        sigma = self.dec_cfg.affinity_sigma

        # pairwise squared distances for each feature
        dist_cpu = squareform(pdist(F[:, 0].reshape(-1, 1), 'sqeuclidean'))
        dist_mem = squareform(pdist(F[:, 1].reshape(-1, 1), 'sqeuclidean'))

        # apply Gaussian Kernel: exp(-d^2 / (2 * sigma^2))
        A_cpu = np.exp(-dist_cpu / (2 * sigma**2))
        A_mem = np.exp(-dist_mem / (2 * sigma**2))

        # convex combination
        A = (alpha * A_cpu) + ((1 - alpha) * A_mem)

        return AffinityMatrix(A)
    
    def build_bundles(self, feature_matrix: FeatureMatrix, affinity_matrix: AffinityMatrix, snapshot: SystemSnapshot)-> ClusteredSnapshot:
        F = feature_matrix.F
        A = affinity_matrix.A
        num_bundles = self.dec_cfg.num_bundles

        # run clustering
        sc = SpectralClustering(n_clusters=num_bundles, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(A)

        pid_to_proc = {p.pid: p for p in snapshot.processes}
        bundle_weights = np.zeros(num_bundles)
        bundle_mem = np.zeros(num_bundles)
        bundle_pids = {i: [] for i in range(num_bundles)}

        for idx, label in enumerate(labels):
            weight = F[idx, 0]
            mem = F[idx, 1]
            pid = feature_matrix.pids[idx]

            bundle_weights[label] += weight
            bundle_mem[label] += mem 
            bundle_pids[label].append(pid)

        # homogeneity enforcement
        theta = self.dec_cfg.homogeneity_threshold
        final_bundles = []

        for i in range(num_bundles):
            members = bundle_pids[i]
            current_bundle_weights = [pid_to_proc[p].cpu_weight for p in members]

            if current_bundle_weights:
                delta_w = max(current_bundle_weights) - min(current_bundle_weights)
            else:
                delta_w = 0

            # Logic for the "Sub-split" 
            # If delta_w > theta, it means this bundle is too "unbalanced."
            # For now, we just proceed, but in a pro version, you'd trigger 
            # another K-means just on this group to break it in two.
            
            # Aggregate stats
            agg_cpu = sum(current_bundle_weights)
            agg_rss = sum(pid_to_proc[p].rss_mb for p in members)

            final_bundles.append(
                Bundle(
                    bundle_id=i,
                    member_pids=members,
                    aggregate_cpu_weight=agg_cpu,
                    aggregate_rss_mb=agg_rss,
                    representative_cmd=f"Bundle_{i}"
                )
            )

        return ClusteredSnapshot(
            bundles=final_bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id
        )