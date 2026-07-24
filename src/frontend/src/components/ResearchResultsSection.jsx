import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  directDepthRows,
  hardCaseRows,
  initialValidationRows,
  mixerComparisonRows,
  scalabilityRows,
} from '../lib/researchResults.js'
import { useMediaQuery } from '../lib/useMediaQuery.js'

const colors = {
  gold: '#F2D985',
  amber: '#C9A24B',
  copper: '#B5693A',
  aqua: '#58C7B6',
  blue: '#78A9FF',
  green: '#8FD16A',
  bone: '#EDE6D9',
  muted: '#8A8072',
}

const chartMargin = { top: 18, right: 22, left: 0, bottom: 8 }
const scalabilityChartMargin = { top: 18, right: 4, left: 0, bottom: 4 }
const scalabilityTooltipOrder = [
  'reference_quality_pct',
  'pipeline_quality_pct',
  'classical_time_seconds',
  'qaoa_time_seconds',
]

function formatNumber(value, digits = 2) {
  return Number(value).toLocaleString('en-US', {
    maximumFractionDigits: digits,
  })
}

function formatPct(value, digits = 3) {
  return `${formatNumber(value, digits)}%`
}

function formatTimeSeconds(value) {
  if (value < 1) return `${formatNumber(value, 4)}s`
  if (value < 60) return `${formatNumber(value, 2)}s`
  return `${formatNumber(value / 60, 2)}m`
}

function ResultTooltip({ active, payload, label, suffixes = {}, tooltipLabels = {}, precision = {}, payloadOrder }) {
  if (!active || !payload?.length) return null
  const orderedPayload = payloadOrder
    ? [...payload].sort((a, b) => payloadOrder.indexOf(a.dataKey) - payloadOrder.indexOf(b.dataKey))
    : payload

  return (
    <div className="chart-tooltip">
      <span className="mono">{label}</span>
      {orderedPayload.map((entry) => {
        const suffix = suffixes[entry.dataKey] || ''
        const value =
          suffix === '%'
            ? formatPct(entry.value, precision[entry.dataKey] ?? 4)
            : suffix === 's'
              ? formatTimeSeconds(entry.value)
              : formatNumber(entry.value, precision[entry.dataKey] ?? 2)
        return (
          <strong className="mono" key={entry.name} style={{ color: entry.color }}>
            {tooltipLabels[entry.dataKey] || entry.name}: {value}
            {suffix && suffix !== '%' && suffix !== 's' ? suffix : ''}
          </strong>
        )
      })}
    </div>
  )
}

function StatCard({ label, value, detail }) {
  return (
    <article className="research-stat glass-surface">
      <span className="mono">{label}</span>
      <strong className="mono">{value}</strong>
      <p>{detail}</p>
    </article>
  )
}

function InsightCard({ label, title, copy }) {
  return (
    <article className="research-insight">
      <span className="mono">{label}</span>
      <h3>{title}</h3>
      <p>{copy}</p>
    </article>
  )
}

function ScalabilityLegend() {
  return (
    <div className="scalability-legend mono" aria-label="Scalability chart legend">
      <span className="legend-classical-reference">Classical Reference</span>
      <span className="legend-hybrid">Hybrid Pipeline</span>
      <span className="legend-classical-time">Classical time</span>
      <span className="legend-qaoa-time">QAOA time</span>
    </div>
  )
}

export function ResearchResultsSection() {
  const isMobile = useMediaQuery('(max-width: 880px)')
  const totalRuns = scalabilityRows.reduce((sum, row) => sum + row.runs, 0)
  const finalScale = scalabilityRows[scalabilityRows.length - 1]
  const scaleChartRows = scalabilityRows.map((row) => ({
    ...row,
    qaoa_time_seconds: row.qaoa_time_s,
    classical_time_seconds: row.classical_time_s,
  }))
  const directChartRows = directDepthRows.map((row) => ({
    ...row,
    timeS: row.timeMs / 1000,
  }))

  return (
    <section className="section-shell research-results-section" id="results" aria-labelledby="research-results-title">
      <div className="research-results-hero">
        <div className="section-heading research-results-heading">
          <p className="eyebrow mono">RESULTS</p>
          <h2 id="research-results-title">What the experiments actually showed</h2>
          <p>
            The strongest result is not a claim of quantum advantage. It is a working hybrid
            pipeline that preserves valid assignments, decomposes large scheduling instances, and
            exposes where QAOA parameter choices matter.
          </p>
        </div>

        <div className="research-stat-grid">
          <StatCard label="scale reached" value={`N=${finalScale.N}`} detail={`${finalScale.num_sub_qubos} fixed-size sub-QUBOs were orchestrated for the largest sweep point.`} />
          <StatCard label="feasibility" value={`${totalRuns}/${totalRuns}`} detail={`Every N=10 to ${finalScale.N} scalability run produced a valid one-hot assignment.`} />
          <StatCard label="final imbalance" value={formatPct(finalScale.pipeline_quality_pct, 4)} detail="Mean normalized load imbalance at N=1000, measured against the average core load." />
          <StatCard label="qaoa time" value={`${formatNumber(finalScale.qaoa_time_s / 60, 1)}m`} detail="Average simulated QAOA time for the N=1000 stress point." />
        </div>
      </div>

      <div className="research-results-grid">
        <article className="research-chart-card research-chart-wide glass-panel">
          <div className="chart-heading research-chart-heading">
            <div>
              <span className="mono">Scalability sweep</span>
              <h3>N=10 to N=1000</h3>
            </div>
            <p>Lower imbalance is better; the classical reference is small but nonzero, and the right axis compares QAOA time with classical time.</p>
          </div>

          <div className="scalability-chart-frame">
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart data={scaleChartRows} margin={scalabilityChartMargin}>
                <defs>
                  <linearGradient id="imbalanceFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={colors.aqua} stopOpacity={0.38} />
                    <stop offset="100%" stopColor={colors.aqua} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
                <XAxis dataKey="N" stroke={colors.muted} tickLine={false} axisLine={false} />
                <YAxis
                  yAxisId="imbalance"
                  stroke={colors.muted}
                  tickLine={false}
                  axisLine={false}
                  width={58}
                  tickFormatter={(value) => formatPct(value, 2)}
                />
                <YAxis
                  yAxisId="time"
                  orientation="right"
                  stroke={colors.muted}
                  tickLine={false}
                  axisLine={false}
                  width={58}
                  domain={[-180, 600]}
                  ticks={[0, 150, 300, 450, 600]}
                  tickFormatter={formatTimeSeconds}
                />
                {!isMobile && (
                  <Tooltip
                    content={
                      <ResultTooltip
                        suffixes={{
                          reference_quality_pct: '%',
                          pipeline_quality_pct: '%',
                          classical_time_seconds: 's',
                          qaoa_time_seconds: 's',
                        }}
                        precision={{ reference_quality_pct: 6 }}
                        tooltipLabels={{
                          pipeline_quality_pct: 'Hybrid Imbalance',
                          reference_quality_pct: 'Classical imbalance',
                        }}
                        payloadOrder={scalabilityTooltipOrder}
                      />
                    }
                  />
                )}
                <Area
                  yAxisId="imbalance"
                  type="monotone"
                  dataKey="pipeline_quality_pct"
                  name="Hybrid Pipeline"
                  stroke={colors.aqua}
                  fill="url(#imbalanceFill)"
                  strokeWidth={2.6}
                  dot={false}
                  isAnimationActive={!isMobile}
                />
                <Line
                  yAxisId="imbalance"
                  type="monotone"
                  dataKey="reference_quality_pct"
                  name="Classical reference"
                  stroke={colors.bone}
                  strokeDasharray="5 5"
                  strokeOpacity={0.72}
                  strokeWidth={1.8}
                  dot={{ r: 2, strokeWidth: 0, fill: colors.bone }}
                  isAnimationActive={!isMobile}
                />
                <Line
                  yAxisId="time"
                  type="monotone"
                  dataKey="classical_time_seconds"
                  name="Classical time"
                  stroke={colors.green}
                  strokeWidth={2.2}
                  dot={{ r: 2 }}
                  isAnimationActive={!isMobile}
                />
                <Line
                  yAxisId="time"
                  type="monotone"
                  dataKey="qaoa_time_seconds"
                  name="QAOA time"
                  stroke={colors.gold}
                  strokeWidth={2.2}
                  dot={{ r: 2 }}
                  isAnimationActive={!isMobile}
                />
              </ComposedChart>
            </ResponsiveContainer>
            <ScalabilityLegend />
          </div>
        </article>

        <article className="research-chart-card glass-panel">
          <div className="chart-heading research-chart-heading">
            <div>
              <span className="mono">Validated base case</span>
              <h3>Small instances hit the exact optimum</h3>
            </div>
            <p>N=2 to N=10 stayed feasible and optimal in the report validation sweep.</p>
          </div>

          <ResponsiveContainer width="100%" height={270}>
            <LineChart data={initialValidationRows} margin={chartMargin}>
              <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
              <XAxis dataKey="N" stroke={colors.muted} tickLine={false} axisLine={false} />
              <YAxis stroke={colors.muted} tickLine={false} axisLine={false} width={46} tickFormatter={(value) => `${value}%`} />
              {!isMobile && <Tooltip content={<ResultTooltip suffixes={{ optimality: '%' }} />} />}
              <Line
                type="monotone"
                dataKey="optimality"
                name="Optimal runs"
                stroke={colors.gold}
                strokeWidth={2.4}
                dot={{ r: 4, strokeWidth: 0, fill: colors.gold }}
                isAnimationActive={!isMobile}
              />
            </LineChart>
          </ResponsiveContainer>
        </article>

        <article className="research-chart-card glass-panel">
          <div className="chart-heading research-chart-heading">
            <div>
              <span className="mono">Direct QAOA tuning</span>
              <h3>Depth recovered the optimum</h3>
            </div>
            <p>For N=8 direct solving, p=2 stayed suboptimal even with 500 optimizer steps; increasing depth to p=3 found the optimum.</p>
          </div>

          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={directChartRows} margin={chartMargin}>
              <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
              <XAxis dataKey="label" stroke={colors.muted} tickLine={false} axisLine={false} interval={0} />
              <YAxis stroke={colors.muted} tickLine={false} axisLine={false} width={42} tickFormatter={(value) => `${formatNumber(value, 0)}s`} />
              {!isMobile && <Tooltip content={<ResultTooltip suffixes={{ timeS: 's' }} />} />}
              <Bar dataKey="timeS" name="Solve time" radius={[4, 4, 0, 0]} isAnimationActive={!isMobile}>
                {directChartRows.map((row) => (
                  <Cell key={row.label} fill={row.optimal ? colors.aqua : colors.copper} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="research-chart-key mono">
            <span className="key-suboptimal">suboptimal</span>
            <span className="key-optimal">optimal</span>
          </div>
        </article>

        <article className="research-chart-card glass-panel">
          <div className="chart-heading research-chart-heading">
            <div>
              <span className="mono">Mixer tradeoff</span>
              <h3>X searched better with tiny top_k; XY stayed valid</h3>
            </div>
            <p>With only the single best candidate, X found more optima; with top_k=10, both mixers recovered the optimum while XY stayed structurally feasible.</p>
          </div>

          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={mixerComparisonRows} margin={chartMargin}>
              <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
              <XAxis dataKey="mixer" stroke={colors.muted} tickLine={false} axisLine={false} />
              <YAxis stroke={colors.muted} tickLine={false} axisLine={false} width={46} tickFormatter={(value) => `${value}%`} />
              {!isMobile && <Tooltip content={<ResultTooltip suffixes={{ topK1Feasible: '%', topK1Optimal: '%', topK10Optimal: '%' }} />} />}
              <Legend iconType="rect" wrapperStyle={{ color: '#8A8072', fontFamily: 'Inter, sans-serif', fontSize: 12 }} />
              <Bar dataKey="topK1Feasible" name="Feasible top_k=1" fill={colors.blue} radius={[4, 4, 0, 0]} isAnimationActive={!isMobile} />
              <Bar dataKey="topK1Optimal" name="Optimal top_k=1" fill={colors.gold} radius={[4, 4, 0, 0]} isAnimationActive={!isMobile} />
              <Bar dataKey="topK10Optimal" name="Optimal top_k=10" fill={colors.aqua} radius={[4, 4, 0, 0]} isAnimationActive={!isMobile} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        <article className="research-chart-card glass-panel">
          <div className="chart-heading research-chart-heading">
            <div>
              <span className="mono">Hard workloads</span>
              <h3>More candidates fixed brittle cases</h3>
            </div>
            <p>Dominant means one heavy process. Depth 1 vs depth 2 is the same workload at different QAOA depths; depth 2 did worse with top_k=3.</p>
          </div>

          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={hardCaseRows} margin={chartMargin}>
              <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
              <XAxis dataKey="scenario" stroke={colors.muted} tickLine={false} axisLine={false} interval={0} />
              <YAxis stroke={colors.muted} tickLine={false} axisLine={false} width={46} tickFormatter={(value) => `${value}%`} />
              {!isMobile && <Tooltip content={<ResultTooltip suffixes={{ topK3: '%', topK10: '%' }} />} />}
              <Legend iconType="rect" wrapperStyle={{ color: '#8A8072', fontFamily: 'Inter, sans-serif', fontSize: 12 }} />
              <Bar dataKey="topK3" name="Optimal top_k=3" fill={colors.copper} radius={[4, 4, 0, 0]} isAnimationActive={!isMobile} />
              <Bar dataKey="topK10" name="Optimal top_k=10" fill={colors.aqua} radius={[4, 4, 0, 0]} isAnimationActive={!isMobile} />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </div>

      <div className="research-insight-grid">
        <InsightCard
          label="01"
          title="Decomposition made larger runs operational"
          copy={`The N=${finalScale.N} run represents ${finalScale.N * 2} global binary variables, but QAOA only handled fixed eight-variable sub-QUBOs.`}
        />
        <InsightCard
          label="02"
          title="Quality claims stay conservative"
          copy="The large-N sweep shows valid, low-imbalance assignments but with not certified global optimality or quantum advantage."
        />
        <InsightCard
          label="03"
          title="Parameter coupling is the main lesson"
          copy="Depth, top_k, and mixer choice change whether the best schedule is visible in the sampled candidate set."
        />
      </div>

      <article className="research-report-panel glass-panel">
        <a className="research-report-cta" href="/report/52585_97.pdf" target="_blank" rel="noreferrer">
          <span className="report-cta-mark mono">PDF</span>
          <div>
            <span className="mono">Full report</span>
            <h3>
              Open the dissertation report
              <strong className="mono">Open PDF</strong>
            </h3>
            <p>Detailed tables, methodology, and the complete discussion of the results.</p>
          </div>
        </a>
      </article>
    </section>
  )
}
