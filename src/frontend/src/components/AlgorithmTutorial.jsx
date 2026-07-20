const quboCells = [
  'hot',
  'penalty',
  'link',
  'quiet',
  'link',
  'penalty',
  'hot',
  'quiet',
  'link',
  'quiet',
  'link',
  'quiet',
  'hot',
  'penalty',
  'link',
  'quiet',
  'link',
  'penalty',
  'hot',
  'quiet',
  'link',
  'quiet',
  'link',
  'quiet',
  'hot',
]

function WorkloadSketch() {
  return (
    <div className="tutorial-glyph workload-glyph" aria-hidden="true">
      {[72, 46, 58, 34].map((width, index) => (
        <span className="workload-row" key={width}>
          <i />
          <b style={{ '--row-width': `${width}%`, '--row-delay': `${index * 0.18}s` }} />
        </span>
      ))}
    </div>
  )
}

function BalanceSketch() {
  const cores = [
    { id: 'c0', load: 68, chips: [48, 30], delay: '0s' },
    { id: 'c1', load: 46, chips: [24, 30], delay: '0.18s' },
    { id: 'c2', load: 58, chips: [36, 28], delay: '0.36s' },
  ]

  return (
    <div className="tutorial-glyph balance-glyph" aria-hidden="true">
      <span className="balance-target">
        <b>avg</b>
      </span>
      {cores.map((core) => (
        <span
          className="balance-core"
          key={core.id}
          style={{ '--core-load': `${core.load}%`, '--core-delay': core.delay }}
        >
          <b>{core.id}</b>
          {core.chips.map((chip, index) => (
            <i key={`${core.id}-${chip}`} style={{ '--chip-width': `${chip}%`, '--chip-delay': `${index * 0.12}s` }} />
          ))}
        </span>
      ))}
    </div>
  )
}

function QuboSketch() {
  return (
    <div className="tutorial-glyph qubo-glyph" aria-hidden="true">
      {quboCells.map((tone, index) => (
        <span className={`qubo-cell ${tone}`} key={index} />
      ))}
    </div>
  )
}

function HamiltonianSketch() {
  return (
    <div className="tutorial-glyph hamiltonian-glyph" aria-hidden="true">
      <span className="ham-token">Q</span>
      <span className="ham-arrow" />
      <span className="ham-token ham-cost">Hc</span>
      <span className="ham-terms">
        <i>Zi</i>
        <i>ZiZj</i>
      </span>
    </div>
  )
}

function QaoaSketch() {
  return (
    <div className="tutorial-glyph qaoa-glyph" aria-hidden="true">
      <span className="qaoa-orbit orbit-a" />
      <span className="qaoa-orbit orbit-b" />
      <span className="qaoa-orbit orbit-c" />
      <span className="qaoa-core" />
      <span className="qaoa-layer layer-cost">C</span>
      <span className="qaoa-layer layer-mixer">M</span>
    </div>
  )
}

function SolutionSketch() {
  const cores = [
    { id: 'c0', load: 69, chips: ['p2', 'p5'], height: 66 },
    { id: 'c1', load: 64, chips: ['p1', 'p4'], height: 61 },
    { id: 'c2', load: 67, chips: ['p0', 'p3'], height: 64 },
  ]

  return (
    <div className="tutorial-glyph solution-glyph" aria-hidden="true">
      {cores.map((core) => (
        <span className="core-lane" key={core.id} style={{ '--load-height': `${core.height}%` }}>
          <b>{core.id}</b>
          <strong>{core.load}%</strong>
          <em />
          {core.chips.map((chip) => (
            <i key={chip}>{chip}</i>
          ))}
        </span>
      ))}
    </div>
  )
}

const tutorialSteps = [
  {
    number: '01',
    label: 'Observe',
    title: 'Processes become weights',
    copy: 'The tracer or preset input is normalized into a snapshot: CPU weight, memory, current core, and process class.',
    visual: <WorkloadSketch />,
  },
  {
    number: '02',
    label: 'Target',
    title: 'Minimize load imbalance',
    copy: 'The objective keeps each core load close to the average target instead of searching blindly through every assignment.',
    visual: <BalanceSketch />,
  },
  {
    number: '03',
    label: 'Encode',
    title: 'Build the QUBO matrix',
    copy: 'Each process-core pair becomes a binary variable. The matrix stores balance terms and assignment penalties.',
    visual: <QuboSketch />,
  },
  {
    number: '04',
    label: 'Translate',
    title: 'QUBO becomes quantum cost',
    copy: 'Binary terms are rewritten with Pauli-Z operators, forming the cost Hamiltonian that QAOA can phase and optimize.',
    visual: <HamiltonianSketch />,
  },
  {
    number: '05',
    label: 'Optimize',
    title: 'QAOA searches low energy',
    copy: 'The QUBO is mapped to a cost Hamiltonian. Cost and mixer layers are tuned by a classical optimizer.',
    visual: <QaoaSketch />,
  },
  {
    number: '06',
    label: 'Decode',
    title: 'Return balanced core lanes',
    copy: 'The selected bitstring is decoded into process-to-core assignments and checked against the final load on each core.',
    visual: <SolutionSketch />,
  },
]

export function AlgorithmTutorial() {
  return (
    <section className="section-shell algorithm-section" id="tutorial" aria-labelledby="tutorial-title">
      <div className="algorithm-header">
        <p className="eyebrow mono">SYSTEM TUTORIAL</p>
        <h2 id="tutorial-title">What happens behind the scheduler</h2>
        <p>
          The system turns process load balancing into an energy minimization problem, then uses a
          hybrid quantum-classical loop to search for a low-energy assignment.
        </p>
      </div>

      <div className="tutorial-steps">
        {tutorialSteps.map((step) => (
          <article className="tutorial-step" key={step.number}>
            <div className="tutorial-step-top">
              <span className="step-number mono">{step.number}</span>
              <span className="step-label mono">{step.label}</span>
            </div>
            {step.visual}
            <h3>{step.title}</h3>
            <p>{step.copy}</p>
          </article>
        ))}
      </div>

      <a className="try-link mono" href="#chamber">
        Try it yourself
        <span aria-hidden="true" />
      </a>
    </section>
  )
}
