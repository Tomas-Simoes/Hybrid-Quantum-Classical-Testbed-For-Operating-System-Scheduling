const stages = [
  {
    label: 'SNAPSHOT',
    title: 'The workload enters as weights.',
    copy:
      'The public run uses a bounded synthetic snapshot, shaped like the live process tracer output. Each process has a CPU weight and a target core count before it enters the optimization path.',
  },
  {
    label: 'QUBO',
    title: 'Assignments become a constrained energy surface.',
    copy:
      'The builder encodes one-hot process-to-core choices into a QUBO. Penalties preserve feasibility, while the objective pushes the system toward balanced load.',
  },
  {
    label: 'DECOMPOSE',
    title: 'Large instances are split into solvable bundles.',
    copy:
      'When the qubit budget would be exceeded, the iterative pipeline partitions the workload into sub-QUBOs and carries load bias forward between solves.',
  },
  {
    label: 'QAOA',
    title: 'The solver samples the best admissible assignment.',
    copy:
      'PennyLane runs QAOA over the encoded Hamiltonian, validates the decoded schedule, and returns energy, feasibility, and assignments for the console.',
  },
]

export function PipelineExplainer() {
  return (
    <section className="section-shell explainer-section" aria-labelledby="explainer-title">
      <div className="section-heading">
        <p className="eyebrow mono">PIPELINE EXPLAINER</p>
        <h2 id="explainer-title">From process weights to a schedule</h2>
      </div>

      <div className="explainer-panel glass-panel">
        {stages.map((stage) => (
          <article className="stage" key={stage.label}>
            <span className="mono stage-label">{stage.label}</span>
            <h3>{stage.title}</h3>
            <p>{stage.copy}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
