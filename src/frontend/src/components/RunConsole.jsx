import { useEffect, useMemo, useRef, useState } from 'react'
import { createRun, getRun } from '../api/client.js'

const POLL_MS = 1500
const DEFAULT_PUBLIC_MAX_N = 6
const DEFAULT_PUBLIC_MAX_CORES = 4
const DEFAULT_PUBLIC_MAX_QUBITS = 16
const DEFAULT_PUBLIC_MAX_QAOA_LAYERS = 3
const DEFAULT_PUBLIC_MAX_QAOA_STEPS = 50
const DEFAULT_PUBLIC_MAX_TOP_K = 32
const DEFAULT_PUBLIC_MAX_QUEUE_SIZE = 25
const ABSOLUTE_PUBLIC_MAX_N = 50
const ABSOLUTE_PUBLIC_MAX_CORES = 4
const ABSOLUTE_PUBLIC_MAX_QUBITS = 16
const ABSOLUTE_PUBLIC_MAX_QAOA_LAYERS = 3
const ABSOLUTE_PUBLIC_MAX_QAOA_STEPS = 50
const ABSOLUTE_PUBLIC_MAX_TOP_K = 32
const ABSOLUTE_PUBLIC_MAX_QUEUE_SIZE = 25
const PUBLIC_MAX_N = publicIntegerEnv(import.meta.env.VITE_PUBLIC_MAX_N, DEFAULT_PUBLIC_MAX_N, ABSOLUTE_PUBLIC_MAX_N)
const PUBLIC_MAX_CORES = publicIntegerEnv(
  import.meta.env.VITE_PUBLIC_MAX_CORES,
  DEFAULT_PUBLIC_MAX_CORES,
  ABSOLUTE_PUBLIC_MAX_CORES,
)
const PUBLIC_MAX_QUBITS = Math.max(
  PUBLIC_MAX_CORES,
  publicIntegerEnv(import.meta.env.VITE_PUBLIC_MAX_QUBITS, DEFAULT_PUBLIC_MAX_QUBITS, ABSOLUTE_PUBLIC_MAX_QUBITS),
)
const PUBLIC_MAX_QAOA_LAYERS = publicIntegerEnv(
  import.meta.env.VITE_PUBLIC_MAX_QAOA_LAYERS,
  DEFAULT_PUBLIC_MAX_QAOA_LAYERS,
  ABSOLUTE_PUBLIC_MAX_QAOA_LAYERS,
)
const PUBLIC_MAX_QAOA_STEPS = publicIntegerEnv(
  import.meta.env.VITE_PUBLIC_MAX_QAOA_STEPS,
  DEFAULT_PUBLIC_MAX_QAOA_STEPS,
  ABSOLUTE_PUBLIC_MAX_QAOA_STEPS,
)
const PUBLIC_MAX_TOP_K = publicIntegerEnv(import.meta.env.VITE_PUBLIC_MAX_TOP_K, DEFAULT_PUBLIC_MAX_TOP_K, ABSOLUTE_PUBLIC_MAX_TOP_K)
const PUBLIC_MAX_QUEUE_SIZE = publicIntegerEnv(
  import.meta.env.VITE_PUBLIC_MAX_QUEUE_SIZE,
  DEFAULT_PUBLIC_MAX_QUEUE_SIZE,
  ABSOLUTE_PUBLIC_MAX_QUEUE_SIZE,
)
const SORTING_STRATEGIES = ['WEIGHT_DESCENDING', 'COUPLING_DESCENDING']
const TUNING_TIPS = [
  ['Conflicts or empty assignments?', 'Raise top K or steps first. With the X mixer, a stronger penalty can also push conflicts out of the best states.'],
  ['Feasible but not optimal?', 'Increase QAOA steps before adding layers. If the curve still stalls, retune initial gamma and beta.'],
  ['Direct run got worse?', 'A larger qubit max can make the full QUBO harder to optimize. Compare it against smaller decomposed runs.'],
  ['Run feels too heavy?', 'Lower processes, steps, layers, or top K while tuning. Increase one setting at a time once the behavior is clear.'],
]
const TERMINAL_RUN_STATUSES = new Set(['done', 'failed', 'error'])
const ACTIVE_RUN_STATUSES = new Set(['queued', 'running'])
const RECENT_TERMINAL_RUN_LIMIT = 5
const WEIGHT_PRECISION = 3
const WEIGHT_SCALE = 10 ** WEIGHT_PRECISION
const WEIGHT_INPUT_STEP = String(1 / WEIGHT_SCALE)
const STATUS_COPY = {
  standby: 'Ready to submit a workload.',
  submitting: 'Submitting the configuration to the backend.',
  queued: 'Backend accepted the job and is waiting for a worker slot.',
  running: 'Backend worker is solving the scheduling instance.',
  done: 'Run completed and results are ready.',
  failed: 'Backend reported that the run failed.',
  error: 'Backend status check failed.',
}

function publicIntegerEnv(value, fallback, maximum = Number.POSITIVE_INFINITY) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric < 1) return fallback
  return Math.min(Math.floor(numeric), maximum)
}

function createNormalizedDescendingWeights(count) {
  const boundedCount = Math.max(1, Number(count) || 1)
  const rawWeights = Array.from({ length: boundedCount }, (_, index) => boundedCount - index)
  const rawTotal = rawWeights.reduce((sum, weight) => sum + weight, 0)
  const exactUnits = rawWeights.map((weight) => (weight / rawTotal) * WEIGHT_SCALE)
  const weightUnits = exactUnits.map((weight) => Math.floor(weight))
  const fractionalOrder = exactUnits
    .map((weight, index) => ({ index, fraction: weight - Math.floor(weight) }))
    .sort((left, right) => right.fraction - left.fraction)
  let remainder = WEIGHT_SCALE - weightUnits.reduce((sum, weight) => sum + weight, 0)

  for (let index = 0; remainder > 0; index += 1, remainder -= 1) {
    weightUnits[fractionalOrder[index % fractionalOrder.length].index] += 1
  }

  return weightUnits.map((weight) => (weight / WEIGHT_SCALE).toFixed(WEIGHT_PRECISION))
}

function sumWeights(weights) {
  return weights.reduce((sum, weight) => sum + Number(weight), 0)
}

const MAX_PUBLIC_WEIGHTS = createNormalizedDescendingWeights(PUBLIC_MAX_N)
const CHAMBER_PRESETS = [
  {
    id: 'effective-n8',
    label: 'Effective',
    evidence: 'Reliable N=6 decomposed baseline, tuned to the public Render limits.',
    config: {
      num_processes: '6',
      num_cores: '2',
      weights: ['0.30', '0.27', '0.22', '0.18', '0.15', '0.11'],
      total_weight: '1.23',
      penalty: '5.0',
      target_load: '',
      layers: '1',
      steps: '25',
      learning_rate: '0.05',
      top_k: '20',
      mixer_type: 'xy',
      init_gamma: '0.5',
      init_beta: '0.5',
      qubit_max: '8',
      io_alpha: '0.5',
      affinity_alpha: '0.8',
      homogeneity_threshold: '0.3',
      zscore_threshold: '1.5',
      sorting_strategy: 'COUPLING_DESCENDING',
      min_rss: '20.0',
      min_cpu: '0.005',
      cpu_interval: '1',
      num_samples: '3',
    },
    rows: [
      ['scope', 'N = 6'],
      ['cores', '2'],
      ['mixer', 'xy'],
      ['penalty', '5.0'],
      ['layers', '1'],
      ['steps', '25'],
      ['top K', '20'],
      ['qubit max', '8'],
    ],
  },
  {
    id: 'dominant-hard',
    label: 'Dominant',
    evidence: 'A hard imbalance case where one process owns most of the weight and top-K is tight.',
    config: {
      num_processes: '6',
      num_cores: '2',
      weights: ['0.01', '0.01', '0.01', '0.01', '0.01', '0.95'],
      total_weight: '1.0',
      penalty: '5.0',
      target_load: '',
      layers: '1',
      steps: '25',
      learning_rate: '0.1',
      top_k: '1',
      mixer_type: 'xy',
      init_gamma: '0.5',
      init_beta: '0.5',
      qubit_max: '6',
      io_alpha: '0.5',
      affinity_alpha: '0.8',
      homogeneity_threshold: '0.3',
      zscore_threshold: '1.5',
      sorting_strategy: 'COUPLING_DESCENDING',
      min_rss: '20.0',
      min_cpu: '0.005',
      cpu_interval: '1',
      num_samples: '3',
    },
    rows: [
      ['scope', 'N = 6'],
      ['cores', '2'],
      ['mixer', 'xy'],
      ['penalty', '5.0'],
      ['layers', '1'],
      ['steps', '25'],
      ['top K', '1'],
      ['qubit max', '6'],
    ],
  },
  {
    id: 'tricore-hard',
    label: 'Tri-core',
    evidence: 'A K=3 iterative decomposition case that stresses mapping and decoding beyond two cores.',
    config: {
      num_processes: '5',
      num_cores: '3',
      weights: ['0.36', '0.27', '0.22', '0.18', '0.12'],
      total_weight: '1.15',
      penalty: '5.0',
      target_load: '',
      layers: '1',
      steps: '25',
      learning_rate: '0.05',
      top_k: '20',
      mixer_type: 'xy',
      init_gamma: '0.5',
      init_beta: '0.5',
      qubit_max: '6',
      io_alpha: '0.5',
      affinity_alpha: '0.8',
      homogeneity_threshold: '0.3',
      zscore_threshold: '1.5',
      sorting_strategy: 'WEIGHT_DESCENDING',
      min_rss: '20.0',
      min_cpu: '0.005',
      cpu_interval: '1',
      num_samples: '3',
    },
    rows: [
      ['scope', 'N = 5'],
      ['cores', '3'],
      ['mixer', 'xy'],
      ['penalty', '5.0'],
      ['layers', '1'],
      ['steps', '25'],
      ['top K', '20'],
      ['qubit max', '6'],
    ],
  },
  {
    id: 'max-public',
    label: 'Max public',
    evidence: 'Largest workload exposed by the active public controls.',
    config: {
      num_processes: String(PUBLIC_MAX_N),
      num_cores: '2',
      weights: MAX_PUBLIC_WEIGHTS,
      total_weight: String(Number(sumWeights(MAX_PUBLIC_WEIGHTS).toFixed(WEIGHT_PRECISION))),
      penalty: '5.0',
      target_load: '',
      layers: String(PUBLIC_MAX_QAOA_LAYERS),
      steps: String(PUBLIC_MAX_QAOA_STEPS),
      learning_rate: '0.05',
      top_k: String(PUBLIC_MAX_TOP_K),
      mixer_type: 'xy',
      init_gamma: '0.5',
      init_beta: '0.5',
      qubit_max: String(PUBLIC_MAX_QUBITS),
      io_alpha: '0.5',
      affinity_alpha: '0.8',
      homogeneity_threshold: '0.3',
      zscore_threshold: '1.5',
      sorting_strategy: 'COUPLING_DESCENDING',
      min_rss: '20.0',
      min_cpu: '0.005',
      cpu_interval: '1',
      num_samples: '3',
    },
    rows: [
      ['scope', `N = ${PUBLIC_MAX_N}`],
      ['cores', '2'],
      ['mixer', 'xy'],
      ['penalty', '5.0'],
      ['layers', String(PUBLIC_MAX_QAOA_LAYERS)],
      ['steps', String(PUBLIC_MAX_QAOA_STEPS)],
      ['top K', String(PUBLIC_MAX_TOP_K)],
      ['qubit max', String(PUBLIC_MAX_QUBITS)],
    ],
  },
]

function cloneConfig(config) {
  return {
    ...config,
    weights: [...config.weights],
  }
}

function normalizeWeights(count, weights) {
  const fallback = count > 0 ? String(Number((1 / count).toFixed(4))) : '1'
  return Array.from({ length: count }, (_, index) => weights[index] ?? fallback)
}

function clampNumber(value, min, max, fallback = min) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallback
  return Math.min(Math.max(numeric, min), max)
}

function clampInteger(value, min, max, fallback = min) {
  return Math.round(clampNumber(value, min, max, fallback))
}

function clampConfigToPublicLimits(config) {
  const numProcesses = clampInteger(config.num_processes, 1, PUBLIC_MAX_N, 1)
  const numCores = clampInteger(config.num_cores, 1, PUBLIC_MAX_CORES, 1)
  const qubitMax = clampInteger(config.qubit_max, numCores, PUBLIC_MAX_QUBITS, numCores)

  return {
    ...config,
    num_processes: String(numProcesses),
    num_cores: String(numCores),
    weights: normalizeWeights(numProcesses, config.weights).slice(0, numProcesses),
    layers: String(clampInteger(config.layers, 1, PUBLIC_MAX_QAOA_LAYERS, 1)),
    steps: String(clampInteger(config.steps, 1, PUBLIC_MAX_QAOA_STEPS, 1)),
    top_k: String(clampInteger(config.top_k, 1, PUBLIC_MAX_TOP_K, 1)),
    qubit_max: String(qubitMax),
  }
}

function numberOrNull(value) {
  return value === '' ? null : Number(value)
}

function runStatusMessage(status, job) {
  if (status !== 'queued') return STATUS_COPY[status] || STATUS_COPY.standby

  const position = Number(job?.queue_position)
  const capacity = Number(job?.queue_capacity) || PUBLIC_MAX_QUEUE_SIZE
  const runningCount = Number(job?.queue_running_count) || 0

  if (!Number.isFinite(position) || position < 1) {
    return `Queued. Backend capacity is ${capacity} pending jobs.`
  }

  if (position === 1 && runningCount > 0) {
    return `Queued ${position}/${capacity}. A backend run is executing; yours starts next.`
  }

  if (position === 1) {
    return `Queued ${position}/${capacity}. Waiting for a backend worker to pick it up.`
  }

  return `Queued ${position}/${capacity}. ${position - 1} queued jobs are ahead of this run.`
}

function mergeJobDetails(existing, job) {
  return existing?.job_id === job.job_id ? { ...existing, ...job } : job
}

function NumberField({ id, label, min, max, step = 'any', value, onChange }) {
  return (
    <label htmlFor={id}>
      <span>{label}</span>
      <input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  )
}

export function RunConsole({ runState, onRunStateChange }) {
  const [config, setConfig] = useState(() => clampConfigToPublicLimits(cloneConfig(CHAMBER_PRESETS[0].config)))
  const [selectedPresetId, setSelectedPresetId] = useState('effective-n8')
  const [error, setError] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [trackedJobs, setTrackedJobs] = useState(() => (runState.job ? [runState.job] : []))
  const pollRefs = useRef(new Map())
  const submitLockedRef = useRef(false)
  const selectedJobIdRef = useRef(runState.jobId)

  const processCount = Math.min(Math.max(Number(config.num_processes) || 1, 1), PUBLIC_MAX_N)
  const activeRunCount = trackedJobs.filter((job) => ACTIVE_RUN_STATUSES.has(job.status)).length
  const isRunLocked = isSubmitting
  const displayedStatus = isSubmitting ? 'submitting' : runState.status === 'idle' ? 'standby' : runState.status
  const statusMessage = runStatusMessage(displayedStatus, runState.job)
  const selectedPreset = CHAMBER_PRESETS.find((preset) => preset.id === selectedPresetId) ?? CHAMBER_PRESETS[0]
  const submitLabel = activeRunCount > 0 ? 'Queue another run' : 'Run scheduler'

  const pipelineLimits = useMemo(
    () =>
      [
        `N<=${PUBLIC_MAX_N}`,
        `cores<=${PUBLIC_MAX_CORES}`,
        `qubits<=${PUBLIC_MAX_QUBITS}`,
        `layers<=${PUBLIC_MAX_QAOA_LAYERS}`,
        `steps<=${PUBLIC_MAX_QAOA_STEPS}`,
        `topK<=${PUBLIC_MAX_TOP_K}`,
      ].join(' · '),
    [],
  )

  useEffect(
    () => () => {
      for (const intervalId of pollRefs.current.values()) {
        window.clearInterval(intervalId)
      }
      pollRefs.current.clear()
    },
    [],
  )

  useEffect(() => {
    selectedJobIdRef.current = runState.jobId
  }, [runState.jobId])

  function updateConfig(name, value) {
    setConfig((current) => {
      if (name === 'num_processes') {
        const nextCount = Math.min(Math.max(Number(value) || 1, 1), PUBLIC_MAX_N)
        return {
          ...current,
          num_processes: String(nextCount),
          weights: normalizeWeights(nextCount, current.weights),
        }
      }
      if (name === 'num_cores') {
        const nextCores = Math.min(Math.max(Number(value) || 1, 1), PUBLIC_MAX_CORES)
        const currentQubitMax = Number(current.qubit_max) || nextCores * 4
        return {
          ...current,
          num_cores: String(nextCores),
          qubit_max: String(clampInteger(Math.max(currentQubitMax, nextCores), nextCores, PUBLIC_MAX_QUBITS, nextCores)),
        }
      }
      if (name === 'layers') {
        return { ...current, layers: String(clampInteger(value, 1, PUBLIC_MAX_QAOA_LAYERS, 1)) }
      }
      if (name === 'steps') {
        return { ...current, steps: String(clampInteger(value, 1, PUBLIC_MAX_QAOA_STEPS, 1)) }
      }
      if (name === 'top_k') {
        return { ...current, top_k: String(clampInteger(value, 1, PUBLIC_MAX_TOP_K, 1)) }
      }
      if (name === 'qubit_max') {
        const nextCores = clampInteger(current.num_cores, 1, PUBLIC_MAX_CORES, 1)
        return { ...current, qubit_max: String(clampInteger(value, nextCores, PUBLIC_MAX_QUBITS, nextCores)) }
      }
      return { ...current, [name]: value }
    })
  }

  function updateWeight(index, value) {
    setConfig((current) => ({
      ...current,
      weights: current.weights.map((weight, weightIndex) => (weightIndex === index ? value : weight)),
    }))
  }

  function selectChamberPreset(preset) {
    setSelectedPresetId(preset.id)
    setConfig(clampConfigToPublicLimits(cloneConfig(preset.config)))
  }

  function buildPayload() {
    const publicConfig = clampConfigToPublicLimits(config)
    const publicProcessCount = Number(publicConfig.num_processes)
    return {
      num_processes: Number(publicConfig.num_processes),
      n_processes: Number(publicConfig.num_processes),
      num_cores: Number(publicConfig.num_cores),
      weights: publicConfig.weights.slice(0, publicProcessCount).map((weight) => Number(weight)),
      total_weight: Number(publicConfig.total_weight),
      penalty: Number(publicConfig.penalty),
      target_load: numberOrNull(publicConfig.target_load),
      layers: Number(publicConfig.layers),
      steps: Number(publicConfig.steps),
      learning_rate: Number(publicConfig.learning_rate),
      top_k: Number(publicConfig.top_k),
      mixer_type: publicConfig.mixer_type,
      init_gamma: Number(publicConfig.init_gamma),
      init_beta: Number(publicConfig.init_beta),
      qubit_max: Number(publicConfig.qubit_max),
      io_alpha: Number(publicConfig.io_alpha),
      affinity_alpha: Number(publicConfig.affinity_alpha),
      homogeneity_threshold: Number(publicConfig.homogeneity_threshold),
      zscore_threshold: Number(publicConfig.zscore_threshold),
      sorting_strategy: publicConfig.sorting_strategy,
      min_rss: Number(publicConfig.min_rss),
      min_cpu: Number(publicConfig.min_cpu),
      cpu_interval: Number(publicConfig.cpu_interval),
      num_samples: Number(publicConfig.num_samples),
    }
  }

  function stopJobPolling(jobId) {
    const intervalId = pollRefs.current.get(jobId)
    if (intervalId) window.clearInterval(intervalId)
    pollRefs.current.delete(jobId)
  }

  function upsertTrackedJob(job) {
    setTrackedJobs((current) => {
      const updated = current.some((item) => item.job_id === job.job_id)
        ? current.map((item) => mergeJobDetails(item, job))
        : [job, ...current]
      const activeJobs = updated.filter((item) => ACTIVE_RUN_STATUSES.has(item.status))
      const terminalJobs = updated.filter((item) => !ACTIVE_RUN_STATUSES.has(item.status))
      return [...activeJobs, ...terminalJobs.slice(0, RECENT_TERMINAL_RUN_LIMIT)]
    })
  }

  function jobToRunState(job, current) {
    return {
      ...current,
      status: job.status,
      jobId: job.job_id,
      job,
      effectiveConfig: job.effective_config || current.effectiveConfig,
    }
  }

  function publishJob(job, { select = false } = {}) {
    upsertTrackedJob(job)
    if (select) {
      selectedJobIdRef.current = job.job_id
      setError(job.status === 'failed' || job.status === 'error' ? jobFailureMessage(job) : null)
    }

    onRunStateChange((current) => {
      if (!select && current.jobId && current.jobId !== job.job_id) return current
      return jobToRunState(mergeJobDetails(current.job, job), current)
    })
  }

  function jobFailureMessage(job) {
    if (job?.status === 'error') return job?.error?.message || 'Backend status check failed.'
    return backendFailureMessage(job)
  }

  function backendFailureMessage(job) {
    const backendMessage = job?.error?.message
    return backendMessage
      ? `Backend failed while running the job: ${backendMessage}`
      : 'Backend failed while running the job. Reduce workload size or retry shortly.'
  }

  async function poll(jobId) {
    try {
      const job = await getRun(jobId)
      publishJob(job)
      if (selectedJobIdRef.current === jobId && job.status !== 'failed') {
        setError(null)
      }

      if (TERMINAL_RUN_STATUSES.has(job.status)) {
        stopJobPolling(jobId)
        if (job.status !== 'done' && selectedJobIdRef.current === jobId) {
          setError(backendFailureMessage(job))
        }
      }
      return job.status
    } catch (pollError) {
      stopJobPolling(jobId)
      const message =
        pollError.status === 404
          ? 'Backend lost the job record before it completed. Submit the run again.'
          : `Backend status check failed: ${pollError.message}`
      const errorJob = {
        job_id: jobId,
        status: 'error',
        error: { type: 'BackendStatusError', message },
      }
      publishJob(errorJob)
      if (selectedJobIdRef.current === jobId) setError(message)
      return 'error'
    }
  }

  function startPolling(jobId) {
    stopJobPolling(jobId)
    pollRefs.current.set(jobId, window.setInterval(() => poll(jobId), POLL_MS))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (submitLockedRef.current || isRunLocked) return
    submitLockedRef.current = true
    setIsSubmitting(true)
    setError(null)

    try {
      setConfig((current) => clampConfigToPublicLimits(current))
      const created = await createRun(buildPayload())
      if (!created?.job_id || !created?.status) {
        throw new Error('Backend accepted the request but did not return a job id.')
      }
      const createdJob = {
        job_id: created.job_id,
        status: created.status,
        queue_position: created.queue_position,
        queue_capacity: created.queue_capacity,
        queue_running_count: created.queue_running_count,
        effective_config: created.effective_config,
        result: null,
        error: null,
      }
      publishJob(createdJob, { select: true })
      setIsSubmitting(false)
      const firstStatus = await poll(created.job_id)
      if (!TERMINAL_RUN_STATUSES.has(firstStatus)) {
        startPolling(created.job_id)
      }
    } catch (runError) {
      const message =
        runError.status === 429
          ? `Submit rejected: ${runError.message}`
          : runError.status === 503
            ? `Submit rejected: ${runError.message}`
            : runError.status === 0
              ? `Submit failed: backend did not respond. ${runError.message}`
              : `Submit failed: ${runError.message}`
      setError(message)
      setIsSubmitting(false)
    } finally {
      submitLockedRef.current = false
    }
  }

  return (
    <section className="section-shell run-console-section" id="chamber" aria-labelledby="run-console-title">
      <div className="section-heading">
        <p className="eyebrow mono">RUN CONSOLE</p>
        <h2 id="run-console-title">
          Welcome to <span>The Chamber</span>
        </h2>
      </div>

      <div className="run-console glass-panel">
        <form className="config-surface" onSubmit={handleSubmit}>
          <div className="config-group">
            <p className="config-heading mono">Workload</p>
            <div className="config-grid">
              <NumberField
                id="process-count"
                label="Processes"
                min="1"
                max={PUBLIC_MAX_N}
                step="1"
                value={config.num_processes}
                onChange={(value) => updateConfig('num_processes', value)}
              />
              <NumberField
                id="core-count"
                label="Cores"
                min="1"
                max={PUBLIC_MAX_CORES}
                step="1"
                value={config.num_cores}
                onChange={(value) => updateConfig('num_cores', value)}
              />
            </div>
            <div className="weights-field">
              <span>Process weights</span>
              <div className="weight-grid">
                {config.weights.slice(0, processCount).map((weight, index) => (
                  <label className="weight-input" htmlFor={`weight-${index}`} key={index}>
                    <span className="mono">p_{index}</span>
                    <input
                      id={`weight-${index}`}
                      type="number"
                      min="0"
                      step={WEIGHT_INPUT_STEP}
                      value={weight}
                      onChange={(event) => updateWeight(index, event.target.value)}
                    />
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="config-group">
            <p className="config-heading mono">QUBO and mixer</p>
            <div className="config-grid">
              <NumberField
                id="penalty"
                label="Penalty"
                min="0.001"
                step="0.001"
                value={config.penalty}
                onChange={(value) => updateConfig('penalty', value)}
              />
              <fieldset>
                <legend>Mixer</legend>
                <div className="segmented-control">
                  <button
                    type="button"
                    className={config.mixer_type === 'xy' ? 'active' : ''}
                    onClick={() => updateConfig('mixer_type', 'xy')}
                  >
                    XY
                  </button>
                  <button
                    type="button"
                    className={config.mixer_type === 'x' ? 'active' : ''}
                    onClick={() => updateConfig('mixer_type', 'x')}
                  >
                    X
                  </button>
                </div>
              </fieldset>
            </div>
          </div>

          <details className="advanced-config">
            <summary className="mono">Advanced configurations</summary>
            <div className="advanced-config-body">
              <div className="config-group">
                <p className="config-heading mono">QAOA</p>
                <div className="config-grid">
                  <NumberField id="layers" label="Layers" min="1" max={PUBLIC_MAX_QAOA_LAYERS} step="1" value={config.layers} onChange={(value) => updateConfig('layers', value)} />
                  <NumberField id="steps" label="Steps" min="1" max={PUBLIC_MAX_QAOA_STEPS} step="1" value={config.steps} onChange={(value) => updateConfig('steps', value)} />
                  <NumberField id="learning-rate" label="Learning rate" min="0.0001" max="1" step="0.0001" value={config.learning_rate} onChange={(value) => updateConfig('learning_rate', value)} />
                  <NumberField id="top-k" label="Top K" min="1" max={PUBLIC_MAX_TOP_K} step="1" value={config.top_k} onChange={(value) => updateConfig('top_k', value)} />
                  <NumberField id="init-gamma" label="Initial gamma" min="0" step="0.001" value={config.init_gamma} onChange={(value) => updateConfig('init_gamma', value)} />
                  <NumberField id="init-beta" label="Initial beta" min="0" step="0.001" value={config.init_beta} onChange={(value) => updateConfig('init_beta', value)} />
                </div>
              </div>

              <div className="config-group">
                <p className="config-heading mono">QUBO target</p>
                <div className="config-grid">
                  <NumberField id="total-weight" label="Total weight" min="0.001" step="0.001" value={config.total_weight} onChange={(value) => updateConfig('total_weight', value)} />
                  <NumberField id="target-load" label="Target load" min="0" step="0.001" value={config.target_load} onChange={(value) => updateConfig('target_load', value)} />
                </div>
              </div>

              <div className="config-group">
                <p className="config-heading mono">Decomposition</p>
                <div className="config-grid">
                  <NumberField id="qubit-max" label="Qubit max" min="1" max={PUBLIC_MAX_QUBITS} step="1" value={config.qubit_max} onChange={(value) => updateConfig('qubit_max', value)} />
                  <NumberField id="io-alpha" label="IO alpha" min="0" max="1" step="0.001" value={config.io_alpha} onChange={(value) => updateConfig('io_alpha', value)} />
                  <NumberField id="affinity-alpha" label="Affinity alpha" min="0" max="1" step="0.001" value={config.affinity_alpha} onChange={(value) => updateConfig('affinity_alpha', value)} />
                  <NumberField id="homogeneity-threshold" label="Homogeneity threshold" min="0" step="0.001" value={config.homogeneity_threshold} onChange={(value) => updateConfig('homogeneity_threshold', value)} />
                  <NumberField id="zscore-threshold" label="Z-score threshold" min="0" step="0.001" value={config.zscore_threshold} onChange={(value) => updateConfig('zscore_threshold', value)} />
                  <label htmlFor="sorting-strategy">
                    <span>Sorting strategy</span>
                    <select
                      id="sorting-strategy"
                      value={config.sorting_strategy}
                      onChange={(event) => updateConfig('sorting_strategy', event.target.value)}
                    >
                      {SORTING_STRATEGIES.map((strategy) => (
                        <option value={strategy} key={strategy}>
                          {strategy}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>

              <div className="config-group">
                <p className="config-heading mono">Tracer filters</p>
                <div className="config-grid">
                  <NumberField id="min-rss" label="Minimum RSS" min="0" step="0.1" value={config.min_rss} onChange={(value) => updateConfig('min_rss', value)} />
                  <NumberField id="min-cpu" label="Minimum CPU" min="0" step="0.001" value={config.min_cpu} onChange={(value) => updateConfig('min_cpu', value)} />
                  <NumberField id="cpu-interval" label="CPU interval" min="1" max="60" step="1" value={config.cpu_interval} onChange={(value) => updateConfig('cpu_interval', value)} />
                  <NumberField id="num-samples" label="Samples" min="1" step="1" value={config.num_samples} onChange={(value) => updateConfig('num_samples', value)} />
                </div>
              </div>
            </div>
          </details>

          <div className="run-action-row">
            <button className="run-button" type="submit" disabled={isRunLocked} aria-busy={isSubmitting}>
              {isSubmitting ? 'Submitting' : submitLabel}
            </button>
            <div className={`status-badge status-${displayedStatus}`} aria-label={`Run status: ${displayedStatus}`}>
              <span className="status-dot" aria-hidden="true" />
              <strong className="mono">{displayedStatus}</strong>
            </div>
          </div>
          <p className="run-status-note mono">{statusMessage}</p>
        </form>

        <div className="telemetry-surface">
          {error && <p className="console-error mono">{error}</p>}
          <div className="pipeline-limit-note">
            <p>
              <span className="mono">Pipeline limits:</span> {pipelineLimits} - Download the unrestricted version on GitHub.
            </p>
          </div>
          <div className="telemetry-panel effective-preset-panel">
            <div className="preset-heading-row">
              <p className="telemetry-heading mono">Chamber presets</p>
            </div>
            <div className="preset-selector" aria-label="Chamber presets">
              {CHAMBER_PRESETS.map((preset) => (
                <button
                  type="button"
                  className={preset.id === selectedPreset.id ? 'active' : ''}
                  aria-pressed={preset.id === selectedPreset.id}
                  onClick={() => selectChamberPreset(preset)}
                  key={preset.id}
                >
                  <span>{preset.label}</span>
                </button>
              ))}
            </div>
            <div className="effective-config-grid">
              {selectedPreset.rows.map(([label, value]) => (
                <div className="effective-config-item" key={label}>
                  <span>{label}</span>
                  <strong className="mono">{value}</strong>
                </div>
              ))}
            </div>
            <p className="preset-evidence">
              {selectedPreset.evidence}
            </p>
          </div>
          <div className="telemetry-panel">
            <p className="telemetry-heading mono">Tuning tips</p>
            <div className="tips-list">
              {TUNING_TIPS.map(([title, copy]) => (
                <article className="tip-item" key={title}>
                  <strong>{title}</strong>
                  <span>{copy}</span>
                </article>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
