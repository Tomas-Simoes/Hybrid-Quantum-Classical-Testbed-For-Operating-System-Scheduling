import { useEffect, useMemo, useRef, useState } from 'react'
import { createRun, getRun } from '../api/client.js'
import { mono } from '../lib/results.js'

const POLL_MS = 1500
const PUBLIC_MAX_N = 10

export function RunConsole({ runState, onRunStateChange, energy, convergence }) {
  const [n, setN] = useState(8)
  const [mixer, setMixer] = useState('xy')
  const [error, setError] = useState(null)
  const [liveTick, setLiveTick] = useState(0)
  const pollRef = useRef(null)

  const isRunning = runState.status === 'queued' || runState.status === 'running'
  const latestCost = convergence.length ? convergence[convergence.length - 1] : null
  const displayIteration = convergence.length || (isRunning ? Math.max(1, liveTick) : 0)
  const displayedStatus = runState.status === 'idle' ? 'standby' : runState.status

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
    if (!isRunning) return undefined
    const interval = window.setInterval(() => setLiveTick((tick) => tick + 1), 900)
    return () => window.clearInterval(interval)
  }, [isRunning])

  useEffect(
    () => () => {
      if (pollRef.current) window.clearInterval(pollRef.current)
    },
    [],
  )

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
    setError(null)
    setLiveTick(0)
    if (pollRef.current) window.clearInterval(pollRef.current)

    try {
      const created = await createRun({
        n_processes: Number(n),
        mixer,
      })
      onRunStateChange({
        status: created.status,
        jobId: created.job_id,
        job: null,
        effectiveConfig: created.effective_config,
      })
      await poll(created.job_id)
      pollRef.current = window.setInterval(() => poll(created.job_id), POLL_MS)
    } catch (runError) {
      setError(`Run rejected: ${runError.message}`)
      onRunStateChange((current) => ({ ...current, status: 'idle' }))
    }
  }

  return (
    <section className="section-shell run-console-section" id="chamber" aria-labelledby="run-console-title">
      <div className="section-heading">
        <p className="eyebrow mono">RUN CONSOLE</p>
        <h2 id="run-console-title">Bounded public execution</h2>
      </div>

      <div className="run-console glass-panel">
        <form className="config-surface" onSubmit={handleSubmit}>
          <label htmlFor="process-count">
            <span>Processes</span>
            <input
              id="process-count"
              type="number"
              min="1"
              max={PUBLIC_MAX_N}
              value={n}
              onChange={(event) => setN(event.target.value)}
            />
          </label>

          <fieldset>
            <legend>Mixer</legend>
            <div className="segmented-control">
              <button
                type="button"
                className={mixer === 'xy' ? 'active' : ''}
                onClick={() => setMixer('xy')}
              >
                XY
              </button>
              <button
                type="button"
                className={mixer === 'x' ? 'active' : ''}
                onClick={() => setMixer('x')}
              >
                X
              </button>
            </div>
          </fieldset>

          <button className="run-button" type="submit" disabled={isRunning}>
            {isRunning ? 'Running' : 'Run scheduler'}
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
          {runState.effectiveConfig && (
            <div className="effective-config">
              <span>effective N</span>
              <strong className="mono">{runState.effectiveConfig.num_processes ?? n}</strong>
              <span>effective mixer</span>
              <strong className="mono">{runState.effectiveConfig.mixer_type ?? mixer}</strong>
            </div>
          )}
          {error && <p className="console-error mono">{error}</p>}
        </div>
      </div>
    </section>
  )
}
