export const scalabilityRows = [
  { N: 10, runs: 20, pipeline_quality_pct: 3.21738947, reference_quality_pct: 0.196945961, total_time_s: 4.73647079, qaoa_time_s: 4.73009719, classical_time_s: 0.00596272185, feasible_pct: 100, num_sub_qubos: 3 },
  { N: 20, runs: 20, pipeline_quality_pct: 0.963290409, reference_quality_pct: 0.000111410452, total_time_s: 17.1959221, qaoa_time_s: 11.4905527, classical_time_s: 5.70448123, feasible_pct: 100, num_sub_qubos: 5 },
  { N: 25, runs: 20, pipeline_quality_pct: 0.292997166, reference_quality_pct: 0.000169965702, total_time_s: 16.6271885, qaoa_time_s: 14.7566089, classical_time_s: 1.86913559, feasible_pct: 100, num_sub_qubos: 7 },
  { N: 30, runs: 20, pipeline_quality_pct: 0.417689183, reference_quality_pct: 0.0000781099831, total_time_s: 19.3433045, qaoa_time_s: 17.1048193, classical_time_s: 2.2366762, feasible_pct: 100, num_sub_qubos: 8 },
  { N: 40, runs: 20, pipeline_quality_pct: 0.140898183, reference_quality_pct: 0.0000684572434, total_time_s: 25.6537907, qaoa_time_s: 22.6380568, classical_time_s: 3.01299892, feasible_pct: 100, num_sub_qubos: 10 },
  { N: 50, runs: 20, pipeline_quality_pct: 0.115381648, reference_quality_pct: 0.0000420505937, total_time_s: 33.0798627, qaoa_time_s: 29.1578506, classical_time_s: 3.91785944, feasible_pct: 100, num_sub_qubos: 13 },
  { N: 65, runs: 12, pipeline_quality_pct: 0.0954408235, reference_quality_pct: 0.0000197249207, total_time_s: 43.314164, qaoa_time_s: 38.102235, classical_time_s: 5.20574303, feasible_pct: 100, num_sub_qubos: 17 },
  { N: 80, runs: 12, pipeline_quality_pct: 0.0562958439, reference_quality_pct: 0.0000218652687, total_time_s: 53.4346536, qaoa_time_s: 46.8427368, classical_time_s: 6.58302031, feasible_pct: 100, num_sub_qubos: 20 },
  { N: 100, runs: 12, pipeline_quality_pct: 0.0329228159, reference_quality_pct: 0.0000105755393, total_time_s: 65.2311122, qaoa_time_s: 57.0097073, classical_time_s: 8.2082395, feasible_pct: 100, num_sub_qubos: 25 },
  { N: 130, runs: 8, pipeline_quality_pct: 0.0221609581, reference_quality_pct: 0.0000115905664, total_time_s: 85.8672272, qaoa_time_s: 74.8496636, classical_time_s: 10.9961718, feasible_pct: 100, num_sub_qubos: 33 },
  { N: 160, runs: 8, pipeline_quality_pct: 0.0199733903, reference_quality_pct: 0.00000640087904, total_time_s: 104.87083, qaoa_time_s: 90.7050434, classical_time_s: 14.1325854, feasible_pct: 100, num_sub_qubos: 40 },
  { N: 200, runs: 8, pipeline_quality_pct: 0.00791750443, reference_quality_pct: 0.00000994324565, total_time_s: 134.357816, qaoa_time_s: 116.153158, classical_time_s: 18.1496733, feasible_pct: 100, num_sub_qubos: 50 },
  { N: 300, runs: 3, pipeline_quality_pct: 0.00192060413, reference_quality_pct: 0.00000924557702, total_time_s: 156.275263, qaoa_time_s: 126.631829, classical_time_s: 29.5242948, feasible_pct: 100, num_sub_qubos: 75 },
  { N: 500, runs: 3, pipeline_quality_pct: 0.000723624937, reference_quality_pct: 0.00000323936535, total_time_s: 324.129182, qaoa_time_s: 267.269621, classical_time_s: 56.4903244, feasible_pct: 100, num_sub_qubos: 125 },
  { N: 750, runs: 3, pipeline_quality_pct: 0.000668592159, reference_quality_pct: 0.00000978921477, total_time_s: 475.269142, qaoa_time_s: 366.895957, classical_time_s: 107.538206, feasible_pct: 100, num_sub_qubos: 188 },
  { N: 1000, runs: 3, pipeline_quality_pct: 0.000395553631, reference_quality_pct: 0.00000766339116, total_time_s: 672.387212, qaoa_time_s: 492.358815, classical_time_s: 178.554544, feasible_pct: 100, num_sub_qubos: 250 },
]

export const initialValidationRows = [
  { N: 2, variables: 4, pipeline: 'Direct', subQubos: 0, optimality: 100, imbalance: 0, timeMs: 896.1 },
  { N: 3, variables: 6, pipeline: 'Direct', subQubos: 0, optimality: 100, imbalance: 0.333, timeMs: 1279.6 },
  { N: 4, variables: 8, pipeline: 'Iterative', subQubos: 2, optimality: 100, imbalance: 0, timeMs: 2401.3 },
  { N: 5, variables: 10, pipeline: 'Iterative', subQubos: 2, optimality: 100, imbalance: 0.2, timeMs: 2869.1 },
  { N: 6, variables: 12, pipeline: 'Iterative', subQubos: 2, optimality: 100, imbalance: 0, timeMs: 3279 },
  { N: 8, variables: 16, pipeline: 'Iterative', subQubos: 3, optimality: 100, imbalance: 0, timeMs: 4202.9 },
  { N: 10, variables: 20, pipeline: 'Iterative', subQubos: 4, optimality: 100, imbalance: 0, timeMs: 5755.9 },
]

export const directDepthRows = [
  { label: 'p=2 / 100', depth: 2, iterations: 100, gap: 0.000591, optimal: 0, timeMs: 8087.1 },
  { label: 'p=2 / 200', depth: 2, iterations: 200, gap: 0.000591, optimal: 0, timeMs: 14141.7 },
  { label: 'p=2 / 500', depth: 2, iterations: 500, gap: 0.001251, optimal: 0, timeMs: 35510.7 },
  { label: 'p=3 / 100', depth: 3, iterations: 100, gap: 0, optimal: 100, timeMs: 10577.5 },
  { label: 'p=4 / 100', depth: 4, iterations: 100, gap: 0, optimal: 100, timeMs: 13701.7 },
]

export const mixerComparisonRows = [
  { mixer: 'X', topK1Feasible: 75, topK1Optimal: 71.7, topK10Optimal: 100, timeMs: 2787 },
  { mixer: 'XY', topK1Feasible: 100, topK1Optimal: 0, topK10Optimal: 100, timeMs: 2200.7 },
]

export const hardCaseRows = [
  { scenario: 'Dominant depth 1', topK3: 96, topK10: 100 },
  { scenario: 'Dominant depth 2', topK3: 36, topK10: 100 },
  { scenario: 'Near-equal depth 1', topK3: 75, topK10: 100 },
  { scenario: 'Near-equal depth 2', topK3: 40, topK10: 100 },
]
