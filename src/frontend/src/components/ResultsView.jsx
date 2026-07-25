import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  balanceReferenceSummary,
  compactNumber,
  coreAssignmentGroups,
  coreLoadComparisonRows,
  optimalityText,
  resultSummary,
  statusText,
} from '../lib/results.js'
import { useMediaQuery } from '../lib/useMediaQuery.js'

export function ResultsView({ job }) {
  const isMobile = useMediaQuery('(max-width: 880px)')

  if (!job?.result) return null

  const groups = coreAssignmentGroups(job)
  const coreComparisons = coreLoadComparisonRows(job)
  const reference = balanceReferenceSummary(job)
  const summary = resultSummary(job)
  const outcomeClass = summary.optimal === true ? 'optimal' : summary.feasible ? 'feasible' : summary.feasible === false ? 'failed' : 'pending'
  const maxLoad = Math.max(...groups.map((group) => group.load), ...groups.map((group) => group.targetLoad), 0)

  return (
    <section className="section-shell results-section" id="run-results" aria-labelledby="results-title">
      <div className="section-heading">
        <p className="eyebrow mono">RESULTS</p>
        <h2 id="results-title">Decoded schedule</h2>
      </div>

      <div className="results-grid">
        <div className="result-summary-panel glass-surface">
          <div className="metric-strip">
            <Metric label="norm. imbalance" value={formatPercent(summary.normalizedLoadImbalance)} featured />
            <Metric label="opt. gap" value={compactNumber(summary.optimalityGap, 4)} />
            <Metric label="load range" value={compactNumber(summary.loadImbalance, 4)} />
            <Metric label="target/core" value={compactNumber(summary.averageCoreLoad, 4)} />
            <Metric label="optimal" value={statusText(summary.optimal)} />
            <Metric label="feasible" value={statusText(summary.feasible)} />
            <Metric label="energy" value={compactNumber(summary.energy, 6)} />
            <Metric label="global best" value={compactNumber(summary.globalEnergy, 6)} />
            <Metric label="variables" value={summary.variables ?? 'pending'} />
            <Metric label="pipeline" value={summary.pipeline} />
            <Metric label="sub-QUBOs" value={summary.subQubos ?? 'pending'} />
            <Metric label="time" value={summary.durationMs === null ? 'pending' : `${compactNumber(summary.durationMs / 1000, 3)}s`} />
          </div>

          <div className={`result-callout result-${outcomeClass}`}>
            <span className="mono">{outcomeLabel(summary)}</span>
            <p>{optimalityText(summary)}</p>
          </div>
        </div>

        <div className="assignment-panel glass-surface">
          <article className="result-chart-card result-assignment-card">
            <div className="chart-heading result-chart-heading">
              <div>
                <span>Core assignment</span>
                <p>Decoded process placement per core.</p>
              </div>
              <strong className="mono">{groups.length} cores</strong>
            </div>
            <div className="core-assignment-map" aria-label="Core assignments">
              {groups.map((group) => (
                <article className="result-core-lane" key={group.core}>
                  <div className="result-core-header">
                    <span className="mono">
                      core<sub>{group.core}</sub>
                    </span>
                    <strong className="mono">{compactNumber(group.load, 4)}</strong>
                  </div>
                  <div className="result-load-track" aria-label={`Core ${group.core} load ${compactNumber(group.load, 4)}`}>
                    <i style={{ width: `${maxLoad ? (group.load / maxLoad) * 100 : 0}%` }} />
                    <b style={{ left: `${maxLoad ? (group.targetLoad / maxLoad) * 100 : 0}%` }} />
                  </div>
                  <div className="result-process-vector">
                    {group.processes.map((process) => (
                      <span className="result-process-chip" key={process.entity}>
                        <strong className="mono">{process.label}</strong>
                        <em className="mono">{compactNumber(process.weight, 3)}</em>
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </article>
        </div>

        <div className="chart-panel glass-surface result-chart-panel">
          <article className="result-chart-card">
            <div className="chart-heading result-chart-heading">
              <div>
                <span>Core totals vs reference</span>
                <p>Summed process weight per core, compared with the average target and classical optimum when available.</p>
              </div>
              <strong className="mono">{reference.hasClassical ? 'checked' : 'no reference'}</strong>
            </div>
            <ResponsiveContainer width="100%" height={252}>
              <BarChart data={coreComparisons} margin={{ top: 14, right: 18, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
                <XAxis dataKey="core" stroke="#8A8072" tickLine={false} axisLine={false} />
                <YAxis stroke="#8A8072" tickLine={false} axisLine={false} width={54} />
                {!isMobile && <Tooltip content={<ComparisonTooltip average={reference.average} />} />}
                {Number.isFinite(reference.average) && (
                  <ReferenceLine y={reference.average} stroke="rgba(237,230,217,0.58)" strokeDasharray="4 5" />
                )}
                {reference.hasClassical && <Bar dataKey="classical" name="classical optimum" fill="#58C7B6" radius={[2, 2, 0, 0]} isAnimationActive={!isMobile} />}
                <Bar dataKey="current" name="decoded schedule" fill="#C9A24B" radius={[2, 2, 0, 0]} isAnimationActive={!isMobile} />
              </BarChart>
            </ResponsiveContainer>
            <div className="result-chart-key">
              <span className="key-current">Decoded</span>
              {reference.hasClassical && <span className="key-classical">Classical optimum</span>}
              <span className="key-average">Average target</span>
            </div>
          </article>
        </div>
      </div>
    </section>
  )
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'pending'
  return `${compactNumber(Number(value) * 100, 4)}%`
}

function Metric({ label, value, featured = false }) {
  return (
    <div>
      <span>{label}</span>
      <strong className={`mono ${featured ? 'gold-number' : ''}`}>{value}</strong>
    </div>
  )
}

function outcomeLabel(summary) {
  if (summary.optimal === true) return 'Optimal result'
  if (summary.optimal === false && summary.feasible === true) return 'Feasible, not optimal'
  if (summary.feasible === true) return 'Feasible result'
  if (summary.feasible === false) return 'Validation failed'
  return 'Pending validation'
}

function ComparisonTooltip({ active, payload, label, average }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span className="mono">{label}</span>
      {payload.map((item) => (
        <strong className="mono" key={item.dataKey}>
          {item.name} {compactNumber(item.value, 6)}
        </strong>
      ))}
      {Number.isFinite(average) && <strong className="mono">average target {compactNumber(average, 6)}</strong>}
    </div>
  )
}
