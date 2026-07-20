import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  assignmentRows,
  convergenceRows,
  extractConfig,
  extractEnergy,
  extractQubo,
  extractValidation,
  mono,
} from '../lib/results.js'

export function ResultsView({ job }) {
  const rows = assignmentRows(job)
  const convergence = convergenceRows(job)
  const validation = extractValidation(job)
  const qubo = extractQubo(job)
  const config = extractConfig(job)
  const energy = extractEnergy(job)

  return (
    <section className="section-shell results-section" aria-labelledby="results-title">
      <div className="section-heading">
        <p className="eyebrow mono">RESULTS</p>
        <h2 id="results-title">Assignments from the last completed run</h2>
      </div>

      {!job?.result ? (
        <div className="empty-state glass-surface">
          <p>The chamber is waiting for a completed run.</p>
        </div>
      ) : (
        <div className="results-grid">
          <div className="assignment-panel glass-surface">
            <div className="metric-strip">
              <div>
                <span>energy</span>
                <strong className="mono gold-number">{mono(energy)}</strong>
              </div>
              <div>
                <span>variables</span>
                <strong className="mono">{qubo?.num_variables ?? 'pending'}</strong>
              </div>
              <div>
                <span>feasible</span>
                <strong className="mono">{String(validation?.valid ?? validation?.is_optimal ?? 'pending')}</strong>
              </div>
              <div>
                <span>N</span>
                <strong className="mono">{config?.num_processes ?? rows.length}</strong>
              </div>
            </div>

            <div className="assignment-list" aria-label="Core assignments">
              {rows.map((row) => (
                <div className="assignment-row" key={row.entity}>
                  <span className="mono">process {row.entity}</span>
                  <span className="assignment-line" aria-hidden="true" />
                  <strong className="mono">core {row.core}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="chart-panel glass-surface">
            <div className="chart-heading">
              <span>Cost over optimization</span>
              <strong className="mono">{convergence.length} samples</strong>
            </div>
            {convergence.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={convergence} margin={{ top: 16, right: 20, left: 0, bottom: 8 }}>
                  <defs>
                    <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#C9A24B" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#C9A24B" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
                  <XAxis dataKey="iteration" stroke="#8A8072" tickLine={false} axisLine={false} />
                  <YAxis stroke="#8A8072" tickLine={false} axisLine={false} width={54} />
                  <Tooltip content={<ConsoleTooltip labelName="iteration" valueName="cost" />} />
                  <Area
                    type="monotone"
                    dataKey="cost"
                    stroke="#C9A24B"
                    fill="url(#costFill)"
                    strokeWidth={2}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="chart-empty">This backend result did not include a convergence curve.</p>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

function ConsoleTooltip({ active, payload, label, labelName, valueName }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span className="mono">
        {labelName} {label}
      </span>
      <strong className="mono">
        {valueName} {mono(payload[0].value)}
      </strong>
    </div>
  )
}
