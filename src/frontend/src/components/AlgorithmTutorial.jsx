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

const processWeights = [
  { id: 'p_0', width: 72 },
  { id: 'p_1', width: 46 },
  { id: 'p_2', width: 58 },
  { id: 'p_3', width: 34 },
]

function WorkloadSketch() {
  return (
    <div className="tutorial-glyph workload-glyph" aria-hidden="true">
      {processWeights.map((process, index) => (
        <span className="workload-row" key={process.id}>
          <i>{process.id}</i>
          <b style={{ '--row-width': `${process.width}%`, '--row-delay': `${index * 0.18}s` }} />
        </span>
      ))}
    </div>
  )
}

function BalanceSketch() {
  const cores = [
    { id: '0', processes: ['p_0', 'p_3'], delay: '0s', duration: '9.6s', laneMin: '88%', laneMid: '95%', laneMax: '100%' },
    { id: '1', processes: ['p_1'], delay: '-2.4s', duration: '10.8s', laneMin: '80%', laneMid: '88%', laneMax: '94%' },
    { id: '2', processes: ['p_2'], delay: '-5.1s', duration: '9.9s', laneMin: '84%', laneMid: '92%', laneMax: '98%' },
  ]

  return (
    <div className="tutorial-glyph balance-glyph" aria-hidden="true">
      <span className="assignment-process-stack">
        {processWeights.map((process, index) => (
          <i key={process.id} style={{ '--process-delay': `${index * 0.08}s` }}>
            {process.id}
          </i>
        ))}
      </span>
      <span className="assignment-flow-lines">
        <i />
        <i />
        <i />
      </span>
      <span className="assignment-core-stack">
        {cores.map((core) => (
          <span
            className="assignment-core-lane"
            key={core.id}
            style={{
              '--core-delay': core.delay,
              '--lane-duration': core.duration,
              '--lane-min': core.laneMin,
              '--lane-mid': core.laneMid,
              '--lane-max': core.laneMax,
            }}
          >
            <b>
              core<sub>{core.id}</sub>
            </b>
            {core.processes.map((process, index) => (
              <i key={`${core.id}-${process}`} style={{ '--chip-delay': `${index * 0.12}s` }}>
                {process}
              </i>
            ))}
          </span>
        ))}
      </span>
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
      <span className="ham-token ham-source">QUBO</span>
      <span className="ham-arrow" />
      <span className="ham-output">
        <span className="ham-token ham-cost">
          H<sub>C</sub>
        </span>
        <span className="ham-plus">+</span>
        <span className="ham-token ham-cost">
          H<sub>M</sub>
        </span>
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
      <span className="qaoa-layer layer-cost">
        H<sub>C</sub>
      </span>
      <span className="qaoa-layer layer-mixer">
        H<sub>M</sub>
      </span>
    </div>
  )
}

function SolutionSketch() {
  const cores = [
    { id: '0', load: 69, chips: ['p_0', 'p_3'], height: 66 },
    { id: '1', load: 64, chips: ['p_1'], height: 61 },
    { id: '2', load: 67, chips: ['p_2'], height: 64 },
  ]

  return (
    <div className="tutorial-glyph solution-glyph" aria-hidden="true">
      {cores.map((core) => (
        <span className="core-lane" key={core.id} style={{ '--load-height': `${core.height}%` }}>
          <b>
            core<sub>{core.id}</sub>
          </b>
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
    copy: 'We extract CPU weight, memory pressure, current core, and process class from the system snapshot.',
    visual: <WorkloadSketch />,
  },
  {
    number: '02',
    label: 'Target',
    title: 'Minimize core imbalance',
    copy: 'The objective penalizes uneven core loads so the solver prefers assignments near the average per-core load.',
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
    copy: 'Binary QUBO terms become Pauli-Z cost terms, forming the Hamiltonian that QAOA can optimize.',
    visual: <HamiltonianSketch />,
  },
  {
    number: '05',
    label: 'Optimize',
    title: 'QAOA searches low energy',
    copy: 'QAOA samples schedules while a classical optimizer tunes circuit angles toward lower-cost assignments.',
    visual: <QaoaSketch />,
  },
  {
    number: '06',
    label: 'Decode',
    title: 'Return balanced assignment',
    copy: 'The best bitstring is decoded into a process-to-core map with the resulting load shown for each core.',
    visual: <SolutionSketch />,
  },
]

export function AlgorithmTutorial({ onNavigate }) {
  function handleTryClick(event) {
    if (!onNavigate) return

    event.preventDefault()
    onNavigate('chamber')
  }

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

      <a className="try-link mono" href="#chamber" onClick={handleTryClick}>
        Try it yourself
        <span aria-hidden="true" />
      </a>
    </section>
  )
}
