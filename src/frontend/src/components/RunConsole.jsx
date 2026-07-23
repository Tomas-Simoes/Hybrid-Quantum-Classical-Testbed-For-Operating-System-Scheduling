import { useEffect, useMemo, useRef, useState } from 'react'
import { createRun, getRun } from '../api/client.js'
import { mono } from '../lib/results.js'

const POLL_MS = 1500
const PUBLIC_MAX_N = Number(import.meta.env.VITE_PUBLIC_MAX_N ?? 6)
const PUBLIC_MAX_CORES = Number(import.meta.env.VITE_PUBLIC_MAX_CORES ?? 4)
const PUBLIC_MAX_QUBITS = Number(import.meta.env.VITE_PUBLIC_MAX_QUBITS ?? 8)
const PUBLIC_MAX_QAOA_LAYERS = Number(import.meta.env.VITE_PUBLIC_MAX_QAOA_LAYERS ?? 1)
const PUBLIC_MAX_QAOA_STEPS = Number(import.meta.env.VITE_PUBLIC_MAX_QAOA_STEPS ?? 25)
const PUBLIC_MAX_TOP_K = Number(import.meta.env.VITE_PUBLIC_MAX_TOP_K ?? 20)
const SORTING_STRATEGIES = ['WEIGHT_DESCENDING', 'COUPLING_DESCENDING', 'CORE_BALANCE']
const TUNING_TIPS = [
  ['Conflicts or empty assignments?', 'Raise top K or steps first. With the X mixer, a stronger penalty can also push conflicts out of the best states.'],
  ['Feasible but not optimal?', 'Increase QAOA steps before adding layers. If the curve still stalls, retune initial gamma and beta.'],
  ['Direct run got worse?', 'A larger qubit max can make the full QUBO harder to optimize. Compare it against smaller decomposed runs.'],
  ['Run feels too heavy?', 'Lower processes, steps, layers, or top K while tuning. Increase one setting at a time once the behavior is clear.'],
]
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
    evidence: 'Largest Render-free workload exposed by the public controls.',
    config: {
      num_processes: '6',
      num_cores: '2',
      weights: ['0.26', '0.21', '0.17', '0.14', '0.12', '0.10'],
      total_weight: '1.0',
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

export function RunConsole({ runState, onRunStateChange, energy, convergence }) {
  const [config, setConfig] = useState(() => clampConfigToPublicLimits(cloneConfig(CHAMBER_PRESETS[0].config)))
  const [selectedPresetId, setSelectedPresetId] = useState('effective-n8')
  const [error, setError] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [liveTick, setLiveTick] = useState(0)
  const pollRef = useRef(null)
  const submitLockedRef = useRef(false)

  const processCount = Math.min(Math.max(Number(config.num_processes) || 1, 1), PUBLIC_MAX_N)
  const isRunning = runState.status === 'queued' || runState.status === 'running'
  const isRunLocked = isSubmitting || isRunning
  const latestCost = convergence.length ? convergence[convergence.length - 1] : null
  const displayIteration = convergence.length || (isRunLocked ? Math.max(1, liveTick) : 0)
  const displayedStatus = isSubmitting ? 'submitting' : runState.status === 'idle' ? 'standby' : runState.status
  const selectedPreset = CHAMBER_PRESETS.find((preset) => preset.id === selectedPresetId) ?? CHAMBER_PRESETS[0]

  const readout = useMemo(
    () => [
      ['status', displayedStatus],
      ['job_id', runState.jobId || 'not submitted'],
      ['iteration', displayIteration || 'pending'],
      ['cost', latestCost === null ? mono(energy) : mono(latestCost)],
    ],
    [displayIteration, displayedStatus, energy, latestCost, runState.jobId],
  )

  useEffect(() => {
    if (!isRunLocked) return undefined
    const interval = window.setInterval(() => setLiveTick((tick) => tick + 1), 900)
    return () => window.clearInterval(interval)
  }, [isRunLocked])

  useEffect(
    () => () => {
      if (pollRef.current) window.clearInterval(pollRef.current)
    },
    [],
  )

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

  async function poll(jobId) {
    const job = await getRun(jobId)
    onRunStateChange((current) => ({
      ...current,
      status: job.status,
      job,
      effectiveConfig: job.effective_config || current.effectiveConfig,
    }))

    if (job.status === 'done' || job.status === 'failed' || job.status === 'error') {
      if (pollRef.current) window.clearInterval(pollRef.current)
      pollRef.current = null
      if (job.status !== 'done') {
        setError(job.error?.message || 'Run failed. Reduce N or retry after the current queue clears.')
      }
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (submitLockedRef.current || isRunLocked) return
    submitLockedRef.current = true
    setIsSubmitting(true)
    setError(null)
    setLiveTick(0)
    if (pollRef.current) window.clearInterval(pollRef.current)

    try {
      setConfig((current) => clampConfigToPublicLimits(current))
      const created = await createRun(buildPayload())
      onRunStateChange({
        status: created.status,
        jobId: created.job_id,
        job: null,
        effectiveConfig: created.effective_config,
      })
      setIsSubmitting(false)
      await poll(created.job_id)
      pollRef.current = window.setInterval(() => poll(created.job_id), POLL_MS)
    } catch (runError) {
      const message =
        runError.status === 429
          ? 'Run rejected: too many clicks in a short window. Wait a moment, then submit once.'
          : runError.status === 503
            ? 'Run rejected: the queue is full. Wait for the current run to finish, then try again.'
            : `Run rejected: ${runError.message}`
      setError(message)
      onRunStateChange((current) => ({ ...current, status: 'idle' }))
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
                      step="0.001"
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

          <button className="run-button" type="submit" disabled={isRunLocked} aria-busy={isRunLocked}>
            {isSubmitting ? 'Submitting' : isRunning ? 'Running' : 'Run scheduler'}
          </button>
          <p className="console-note">
            The server clamps public inputs before the scheduler receives them.
          </p>
        </form>

        <div className="telemetry-surface">
          {readout.map(([label, value]) => (
            <div className="telemetry-row" key={label}>
              <span>{label}</span>
              <strong className="mono">{value}</strong>
            </div>
          ))}
          {error && <p className="console-error mono">{error}</p>}
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
