import { useEffect, useMemo, useState } from 'react'
import {
  Brush,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getScalability } from '../api/client.js'
import { mono } from '../lib/results.js'

const ranges = [
  ['all', 'All'],
  ['small', 'N <= 100'],
  ['large', 'N >= 100'],
]

export function ScalabilityChart() {
  const [rows, setRows] = useState([])
  const [range, setRange] = useState('all')
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    getScalability()
      .then((data) => {
        if (alive) setRows(data.rows || [])
      })
      .catch((scalabilityError) => {
        if (alive) setError(`Scalability data unavailable: ${scalabilityError.message}`)
      })
    return () => {
      alive = false
    }
  }, [])

  const filtered = useMemo(() => {
    if (range === 'small') return rows.filter((row) => row.N <= 100)
    if (range === 'large') return rows.filter((row) => row.N >= 100)
    return rows
  }, [range, rows])

  const totalRuns = rows.reduce((sum, row) => sum + Number(row.runs || 0), 0)

  return (
    <section className="section-shell scalability-section" aria-labelledby="scalability-title">
      <div className="section-heading">
        <p className="eyebrow mono">SCALABILITY</p>
        <h2 id="scalability-title">Precomputed sweep from N=10 to N=1000</h2>
      </div>

      <div className="scalability-panel glass-panel hex-texture-panel">
        <div className="chart-toolbar">
          <div>
            <span>resolved runs</span>
            <strong className="mono gold-number">{totalRuns || 'pending'}</strong>
          </div>
          <div className="segmented-control">
            {ranges.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={range === key ? 'active' : ''}
                onClick={() => setRange(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {error ? (
          <p className="console-error mono">{error}</p>
        ) : (
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={filtered} margin={{ top: 24, right: 26, left: 0, bottom: 20 }}>
              <CartesianGrid stroke="rgba(237,230,217,0.08)" vertical={false} />
              <XAxis dataKey="N" stroke="#8A8072" tickLine={false} axisLine={false} />
              <YAxis
                yAxisId="quality"
                stroke="#8A8072"
                tickLine={false}
                axisLine={false}
                width={54}
              />
              <YAxis
                yAxisId="time"
                orientation="right"
                stroke="#8A8072"
                tickLine={false}
                axisLine={false}
                width={64}
              />
              <Tooltip content={<ScalabilityTooltip />} />
              <Legend iconType="plainline" wrapperStyle={{ color: '#8A8072', fontFamily: 'Inter, sans-serif' }} />
              <Line
                yAxisId="quality"
                type="monotone"
                dataKey="pipeline_quality_pct"
                name="pipeline quality %"
                stroke="#C9A24B"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                yAxisId="quality"
                type="monotone"
                dataKey="reference_quality_pct"
                name="reference quality %"
                stroke="#B5693A"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="time"
                type="monotone"
                dataKey="qaoa_time_s"
                name="QAOA time s"
                stroke="#EDE6D9"
                strokeOpacity={0.58}
                strokeWidth={1.6}
                dot={false}
              />
              <Brush dataKey="N" height={28} stroke="#C9A24B" fill="rgba(23,19,15,0.88)" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  )
}

function ScalabilityTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span className="mono">N {label}</span>
      {payload.map((entry) => (
        <strong className="mono" key={entry.name} style={{ color: entry.color }}>
          {entry.name}: {mono(entry.value, 6)}
        </strong>
      ))}
    </div>
  )
}
