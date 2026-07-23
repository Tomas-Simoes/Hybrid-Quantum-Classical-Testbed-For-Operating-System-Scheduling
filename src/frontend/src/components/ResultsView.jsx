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
  balanceComparisonRows,
  balanceReferenceSummary,
  compactNumber,
  coreAssignmentGroups,
  coreLoadComparisonRows,
  optimalityText,
  resultSummary,
  statusText,
} from '../lib/results.js'

export function ResultsView({ job }) {
  if (!job?.result) return null

  const groups = coreAssignmentGroups(job)
  const coreComparisons = coreLoadComparisonRows(job)
  const balanceRows = balanceComparisonRows(job)
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
        <div className="assignment-panel glass-surface">
          <div className="metric-strip">
            <Metric label="norm. imbalance" value={formatPercent(summary.normalizedLoadImbalance)} featured />
            <Metric label="regret" value={compactNumber(summary.objectiveRegret, 6)} />
            <Metric label="optimal" value={statusText(summary.optimal)} />
            <Metric label="feasible" value={statusText(summary.feasible)} />
            <Metric label="objective" value={compactNumber(summary.balanceObjective, 6)} />
            <Metric label="reference obj." value={compactNumber(summary.classicalBalanceObjective, 6)} />
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
                <Tooltip content={<ComparisonTooltip average={reference.average} />} />
                {Number.isFinite(reference.average) && (
                  <ReferenceLine y={reference.average} stroke="rgba(237,230,217,0.58)" strokeDasharray="4 5" />
                )}
                {reference.hasClassical && <Bar dataKey="classical" name="classical optimum" fill="#58C7B6" radius={[2, 2, 0, 0]} />}
                <Bar dataKey="current" name="decoded schedule" fill="#C9A24B" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="result-chart-key">
              <span className="key-current">Decoded</span>
              {reference.hasClassical && <span className="key-classical">Classical optimum</span>}
              <span className="key-average">Average target</span>
            </div>
          </article>

          <div className="result-chart-split">
            <article className="result-chart-card">
              <div className="chart-heading result-chart-heading compact">
                <div>
                  <span>Balance quality</span>
                  <p>Lower is better. These are the report metrics computed from core totals.</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={178}>
                <BarChart data={balanceRows} layout="vertical" margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
                  <XAxis type="number" stroke="#8A8072" tickLine={false} axisLine={false} />
                  <YAxis dataKey="metric" type="category" stroke="#8A8072" tickLine={false} axisLine={false} width={104} />
                  <Tooltip content={<ComparisonTooltip />} />
                  {reference.hasClassical && <Bar dataKey="classical" name="classical optimum" fill="#58C7B6" radius={[0, 2, 2, 0]} />}
                  <Bar dataKey="current" name="decoded schedule" fill="#C9A24B" radius={[0, 2, 2, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </article>

            <article className="result-chart-card result-reference-card">
              <div className="chart-heading result-chart-heading compact">
                <div>
                  <span>Reference check</span>
                  <p>{reference.hasClassical ? 'Difference from the certified brute-force result.' : 'No brute-force reference was returned for this size.'}</p>
                </div>
              </div>
              <div className="result-reference-grid">
                <ReferenceMetric label="decoded imbalance" value={formatPercent(summary.normalizedLoadImbalance)} />
                <ReferenceMetric label="reference imbalance" value={formatPercent(summary.classicalNormalizedLoadImbalance)} />
                <ReferenceMetric label="objective regret" value={compactNumber(summary.objectiveRegret, 6)} />
                <ReferenceMetric label="excess imbalance" value={formatPercent(summary.excessNormalizedLoadImbalance)} />
              </div>
              <p className="result-reference-note">
                {reference.hasClassical
                  ? 'Zero regret means the decoded assignment matches the reference balance objective.'
                  : 'For larger runs, this panel will still show decoded balance but cannot certify optimality.'}
              </p>
            </article>
          </div>
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

function ReferenceMetric({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong className="mono">{value}</strong>
    </div>
  )
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
