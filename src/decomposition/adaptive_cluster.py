import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering

from data_contracts import AffinityMatrix, Bundle, ClusteredSnapshot, DecompositorConfig, FeatureMatrix, SystemSnapshot

class AdaptiveCluster:
    def __init__(self, decompositor_cfg: DecompositorConfig):
        self.dec_cfg = decompositor_cfg

    def decompose(self, snapshot: SystemSnapshot):
        rt_procs, normal_procs = self._separate_rt_processes(snapshot)

        if not normal_procs:
            # only RT processes => trivial decomposition
            return self._trivial_decomposition(snapshot, rt_procs)

        n = len(normal_procs)
        n_bundles = self.dec_cfg.num_bundles(n) 

        # no clustering needed
        if n_bundles >= n or n==1:
            return self._trivial_decomposition(snapshot)
        
        normal_snapshot = SystemSnapshot(
            timestamp=time.time(),
            snapshot_id=snapshot.snapshot_id,
            processes=normal_procs,
            num_cores=snapshot.num_cores,
            total_ram_mb=snapshot.total_ram_mb
        )

        feature_matrix = self.build_feature_matrix(normal_snapshot)
        affinity_matrix = self.build_affinity_matrix(feature_matrix)
        
        try:
            clustered_snapshot = self.build_bundles(feature_matrix, affinity_matrix, normal_snapshot, n_bundles)
        except Exception as e:
            import warnings
            warnings.warn(f"SpectralClustering failed: ({e}). Using K-Means fallback.")
            clustered_snapshot = self._kmeans_fallback(feature_matrix, normal_snapshot, n_bundles)

        clustered_snapshot.num_cores = self.dec_cfg.num_cores
        clustered_snapshot.rt_procs = rt_procs
        return clustered_snapshot

    def build_feature_matrix(self, snapshot: SystemSnapshot) -> FeatureMatrix:
        """
        This method builds a feature matrix F, where F[i,j] is the feature j of process i
        
        1. Firstly it calculates the process true demand if it spent significant
        time waiting for I/O: w_eff = w_i * (1 - io_alpha * l_i)  
        
        If l_i == 0 (pure CPU-bound): w_eff = w_i;   If l_i == 1 (always waiting I/O): w_eff = 0; 
        
        2. Since these values are in different units and scales we do a Z-Score normalization which maps
        each column to zero mean and unit variance

        3.
        """
        raw_features = []
        pids = []
        w_effs = []

        for p in snapshot.processes:
            # I/O correction to CPU weight
            w_eff = p.cpu_weight * (1 - self.dec_cfg.io_alpha * p.io_wait_ratio)

            raw_features.append([w_eff, p.rss_mb])
            pids.append(p.pid)
            w_effs.append(w_eff)
        
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
            w_eff=np.array(w_effs)
        )

    def build_affinity_matrix(self, feature_matrix: FeatureMatrix) -> AffinityMatrix:
        """
        This method builds a affinity matrix A, where A[i,j] is the similarity between process i and process j
        It knows nothing about clustering, only answer "how similar is this two processes"

        1. Computes squared distances matrices for CPU (w_eff) and Memory (RSS): d^2 = (x_i - x_j)^2
        
        2. Then it transofrms distances into affinities using an RBF kernel: A[i,j] = exp(-d^2 / (2 * sigma^2))
        Sigma is adaptively set based on the median distance to ensure the kernel scales with the data destribution

        3. Convex Combination: It merges the feature-specific affinities into a single matrix using affinity_alpha
        to express which feature matters more for grouping: A_final = alpha * A_cpu + (1- alpha) * A_mem

        At alpha=1.0 clustering is purely CPU-driven, alpha=0.5 CPU and memory contribute equally
        """
        F = feature_matrix.F_norm               # F_norm has columns: [0: CPU, 1: RSS] (IO ratio is embeded in CPU already) 
        alpha = self.dec_cfg.affinity_alpha

        cpu_col = F[:,0].reshape(-1, 1)
        mem_col = F[:,1].reshape(-1, 1)

        # pairwise squared distances for each feature
        dist_cpu = squareform(pdist(cpu_col, 'sqeuclidean'))
        dist_mem = squareform(pdist(mem_col, 'sqeuclidean'))

        def adaptive_sigma(dist_matrix: np.ndarray) -> float:
            upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            median_dist = np.median(upper)
            return float(np.sqrt(median_dist / 2.0)) if median_dist > 1e-9 else 1.0
        
        sigma_cpu = adaptive_sigma(dist_cpu)
        sigma_mem = adaptive_sigma(dist_mem)
        
        # apply Gaussian Kernel: exp(-d^2 / (2 * sigma^2))
        A_cpu = np.exp(-dist_cpu / (2 * sigma_cpu**2))
        A_mem = np.exp(-dist_mem / (2 * sigma_mem**2))

        # convex combination
        A = (alpha * A_cpu) + ((1 - alpha) * A_mem)

        return AffinityMatrix(A)
    
    def build_bundles(self, feature_matrix: FeatureMatrix, affinity_matrix: AffinityMatrix, snapshot: SystemSnapshot, num_bundles: int)-> ClusteredSnapshot:
        """
        Takes the data and decides which bundles to cluster

        1) Does a SpectralClustering which returns our bundles B
        2) After clustering, each bundle needs to pass quality control:
            - Intra-bundle heterogeneity: This flags bundles where the internal spread of weights is too large.
            - Inter-bundle weight outlier: This flags bundles that are statistically heavy relative to other bundles
            If any of these conditions are met && bundle is not singleton then we split the bundle
        """
        
        F = feature_matrix.F
        A = affinity_matrix.A

        # run clustering
        sc = SpectralClustering(n_clusters=num_bundles, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(A)

        pid_to_proc = {p.pid: p for p in snapshot.processes}
        pid_to_weff = dict(zip(feature_matrix.pids, feature_matrix.w_eff))

        bundle_agg_cpu  = np.zeros(num_bundles)
        bundle_agg_rss  = np.zeros(num_bundles)
        bundle_pids = {i: [] for i in range(num_bundles)}

        for idx, label in enumerate(labels):
            pid = feature_matrix.pids[idx]
            bundle_pids[label].append(pid)
            bundle_agg_cpu[label] += pid_to_weff[pid]
            bundle_agg_rss[label] += pid_to_proc[pid].rss_mb

        bundle_weight_mean = np.mean(bundle_agg_cpu)
        bundle_weight_std = np.std(bundle_agg_cpu)
        
        if bundle_weight_std < 1e-9:
            bundle_weight_std = 1.0  # all bundles equal weight, no outliers
        
        W_total = sum(pid_to_weff.values())
        L_avg = W_total / snapshot.num_cores
        
        final_bundles = []
        next_bundle_id = 0  
        for i in range(num_bundles):
            members = bundle_pids[i]                                        # list of PIDs in this bundle
            current_bundle_weights = [pid_to_weff[p]  for p in members]     # w_eff per member

            if not members:
                continue

            # 1. intra-bundle homogeneity (spread within this bundle)
            delta_w = max(current_bundle_weights) - min(current_bundle_weights)
            bundle_mean = np.mean(current_bundle_weights)
            if bundle_mean > 1e-9:
                cv = delta_w / bundle_mean  # coefficient of variation
                is_heterogeneous = cv > self.dec_cfg.homogeneity_threshold
                
            # 2. bundle weight z-score (bundle vs all other bundles)
            z_score = (bundle_agg_cpu[i] - bundle_weight_mean) / bundle_weight_std
            is_heavy_outlier = z_score > self.dec_cfg.zscore_threshold

            # 3. check if they are a "memory monster"
            mem_cap = snapshot.total_ram_mb / snapshot.num_cores
            is_mem_heavy = bundle_agg_rss[i] > (mem_cap * 0.4)
            print(f"memcap: {mem_cap} bundle rss: {bundle_agg_rss[i]}")
            if is_mem_heavy:
                print("found a mem heavy:", i)

            if(is_heterogeneous or is_heavy_outlier or is_mem_heavy) and len(members) > 1:
                # split bundle into two via K-Means
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
                        representative_cmd=f"Bundle_{next_bundle_id}"
                    )
                )
                next_bundle_id += 1


        return ClusteredSnapshot(
            bundles=final_bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id
        )

    def _split_bundle(self, members, pid_to_weff, pid_to_proc,
                  start_id, L_avg, depth=0) -> list[Bundle]:
        MAX_DEPTH = 3
        weights = np.array([pid_to_weff[p] for p in members]).reshape(-1, 1)

        # Base case — can't or shouldn't split further
        if len(members) == 1 or depth >= MAX_DEPTH or np.allclose(weights, weights[0], atol=1e-5):
            return [Bundle(
                bundle_id=start_id,
                member_pids=members,
                aggregate_cpu_weight=float(weights.sum()),
                aggregate_rss_mb=sum(pid_to_proc[p].rss_mb for p in members),
                representative_cmd=f"Bundle_{start_id}"
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

            # This sub-group is still too heavy AND splittable — recurse
            if sub_agg > L_avg * 1.5 and len(sub_members) > 1:
                sub_result = self._split_bundle(
                    sub_members, pid_to_weff, pid_to_proc,
                    start_id + id_offset, L_avg, depth + 1  # depth increases each time
                )
            else:
                sub_result = [Bundle(
                    bundle_id=start_id + id_offset,
                    member_pids=sub_members,
                    aggregate_cpu_weight=sub_agg,
                    aggregate_rss_mb=sum(pid_to_proc[p].rss_mb for p in sub_members),
                    representative_cmd=f"Bundle_{start_id + id_offset}"
                )]

            result.extend(sub_result)
            id_offset += len(sub_result)

        return result
    
    def _separate_rt_processes(self, snapshot: SystemSnapshot) -> tuple[list, list]:
        """
        RT processes are not variables, they are constants since the OS has already decided which core
        is going to. Keeping them in clustering would corrupt the affinity matrix
        """
        rt_procs    = [p for p in snapshot.processes if p.priority_class == "RT"]
        be_procs    = [p for p in snapshot.processes if p.priority_class != "RT"]
        return rt_procs, be_procs
    
    def _kmeans_fallback(self, feature_matrix: FeatureMatrix, snapshot: SystemSnapshot, n_bundles: int) -> ClusteredSnapshot:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_bundles, random_state=42, n_init=10)
        labels = km.fit_predict(feature_matrix.F_norm)

        pid_to_proc = {p.pid: p for p in snapshot.processes}
        pid_to_weff = dict(zip(feature_matrix.pids, feature_matrix.w_eff))

        bundle_agg_cpu = np.zeros(n_bundles)
        bundle_agg_rss = np.zeros(n_bundles)
        bundle_pids    = {i: [] for i in range(n_bundles)}

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
                representative_cmd=f"Bundle_{i}_fallback"
            )
            for i in range(n_bundles) if bundle_pids[i]
        ]

        return ClusteredSnapshot(
            bundles=bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id
        )
    def _trivial_decomposition(self, snapshot: SystemSnapshot, 
                            procs: list | None = None) -> ClusteredSnapshot:
        """Each process becomes its own singleton bundle. No clustering needed."""
        target = procs if procs is not None else snapshot.processes
        bundles = [
            Bundle(
                bundle_id=i,
                member_pids=[p.pid],
                aggregate_cpu_weight=p.cpu_weight,
                aggregate_rss_mb=p.rss_mb,
                representative_cmd=f"Bundle_{i}_singleton"
            )
            for i, p in enumerate(target)
        ]
        return ClusteredSnapshot(
            bundles=bundles,
            num_cores=snapshot.num_cores,
            source_snapshot_id=snapshot.snapshot_id
        )