function hexPoints(cx, cy, radius) {
  return Array.from({ length: 6 }, (_, index) => {
    const angle = Math.PI / 6 + (Math.PI * 2 * index) / 6
    return `${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`
  }).join(' ')
}

const matrixCells = Array.from({ length: 16 }, (_, index) => ({
  x: -15 + (index % 4) * 8,
  y: -15 + Math.floor(index / 4) * 8,
  step: index % 4,
}))

const solutionCells = [
  { x: 0, y: 0, center: true },
  { x: 15.6, y: 0 },
  { x: 7.8, y: 13.5 },
  { x: -7.8, y: 13.5 },
  { x: -15.6, y: 0 },
  { x: -7.8, y: -13.5 },
  { x: 7.8, y: -13.5 },
]

function ProcessIcon() {
  return (
    <svg className="pipeline-icon" viewBox="0 0 62 46" aria-hidden="true">
      <circle className="sigil-halo" cx="31" cy="23" r="22" />
      <g className="process-stack">
        <rect className="process-card sigil-delay-zero" x="13" y="10" width="36" height="7" />
        <rect className="process-card sigil-delay-one" x="13" y="21" width="28" height="7" />
        <rect className="process-card sigil-delay-two" x="13" y="32" width="33" height="7" />
        <circle className="process-dot sigil-delay-zero" cx="18" cy="13.5" r="1.7" />
        <circle className="process-dot sigil-delay-one" cx="18" cy="24.5" r="1.7" />
        <circle className="process-dot sigil-delay-two" cx="18" cy="35.5" r="1.7" />
      </g>
    </svg>
  )
}

function MatrixIcon() {
  return (
    <svg className="pipeline-icon" viewBox="0 0 62 46" aria-hidden="true">
      <circle className="sigil-halo" cx="31" cy="23" r="23" />
      <g className="matrix-grid" transform="translate(31 23)">
        <rect className="matrix-frame" x="-19" y="-19" width="38" height="38" />
        {matrixCells.map((cell, index) => (
          <rect
            className={`matrix-cell sigil-delay-${cell.step}`}
            key={index}
            x={cell.x}
            y={cell.y}
            width="6"
            height="6"
          />
        ))}
      </g>
    </svg>
  )
}

function QaoaIcon() {
  return (
    <svg className="pipeline-icon" viewBox="0 0 62 46" aria-hidden="true">
      <circle className="sigil-halo" cx="31" cy="23" r="24" />
      <g transform="translate(31 23)">
        <ellipse className="quantum-orbit sigil-delay-zero" rx="25" ry="7" />
        <ellipse className="quantum-orbit sigil-delay-one" rx="25" ry="7" transform="rotate(58)" />
        <ellipse className="quantum-orbit sigil-delay-two" rx="25" ry="7" transform="rotate(-58)" />
        <circle className="quantum-core" r="4.6" />
      </g>
    </svg>
  )
}

function SolutionIcon() {
  return (
    <svg className="pipeline-icon" viewBox="0 0 62 46" aria-hidden="true">
      <circle className="sigil-halo" cx="31" cy="23" r="24" />
      <g transform="translate(31 23)">
        {solutionCells.map((cell, index) => (
          <polygon
            className={`solution-cell ${cell.center ? 'solution-center' : `sigil-delay-${index % 4}`}`}
            key={index}
            points={hexPoints(cell.x * 0.92, cell.y * 0.92, 8.4)}
          />
        ))}
        <path className="solution-check" d="M-9 1 L-3 7 L11 -8" />
      </g>
    </svg>
  )
}

const pipelineSteps = [
  { label: 'processes', icon: <ProcessIcon /> },
  { label: 'matrix QUBO', icon: <MatrixIcon /> },
  { label: 'QAOA', icon: <QaoaIcon /> },
  { label: 'solution', icon: <SolutionIcon /> },
]

const connectorPaths = [
  {
    curve: 'M2 15 C26 10 66 11 96 14',
    head: 'M90 10 L97 14 L90 18',
  },
  {
    curve: 'M2 14 C30 11 66 17 96 13',
    head: 'M90 9 L97 13 L90 17',
  },
  {
    curve: 'M2 13 C30 18 66 17 96 12',
    head: 'M90 8 L97 12 L90 16',
  },
]

function PipelineFlow() {
  return (
    <div className="pipeline-flow" aria-hidden="true">
      {pipelineSteps.map((step, index) => {
        const connector = connectorPaths[index]

        return (
          <div className={`pipeline-step pipeline-stage-${index}`} key={step.label}>
            {step.icon}
            <span className="pipeline-label mono">{step.label}</span>
            {connector ? (
              <span className={`pipeline-connector connector-stage-${index}`}>
                <svg className="connector-line" viewBox="0 0 100 26" preserveAspectRatio="none" aria-hidden="true">
                  <path className="connector-curve" pathLength="1" d={connector.curve} />
                  <path className="connector-head" pathLength="1" d={connector.head} />
                </svg>
              </span>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

export function Hero({ lattice }) {
  return (
    <section className="hero-section" id="top" aria-labelledby="hero-title">
      <div className="hero-visual" aria-hidden="true">
        {lattice}
      </div>
      <div className="hero-copy">
        <h1 id="hero-title">The Chamber</h1>
        <p className="hero-subtitle">Hybrid Quantum-Classical Scheduling</p>
        <div
          className="hero-pipeline-art hero-pipeline-support"
          role="img"
          aria-label="Processes flow into a QUBO matrix, then QAOA, then a solution."
        >
          <PipelineFlow />
        </div>
      </div>
    </section>
  )
}
