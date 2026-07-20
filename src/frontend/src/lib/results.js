export function unwrapJob(job) {
  return job?.result || null
}

export function unwrapPipelineOutput(job) {
  return unwrapJob(job)?.result || null
}

export function extractAssignments(job) {
  const output = unwrapPipelineOutput(job)
  if (!output) return null
  return output.final_assignments || output.result?.decoded_assignments || null
}

export function extractConvergence(job) {
  const output = unwrapPipelineOutput(job)
  if (!output) return []

  const direct = output.result?.convergence_curve
  if (Array.isArray(direct) && direct.length) return direct.map(Number)

  const global = output.global_result?.convergence_curve
  if (Array.isArray(global) && global.length) return global.map(Number)

  const subRuns = output.solver_results || []
  return subRuns.flatMap((run) =>
    Array.isArray(run.convergence_curve) ? run.convergence_curve.map(Number) : [],
  )
}

export function extractEnergy(job) {
  const output = unwrapPipelineOutput(job)
  return output?.result?.energy ?? output?.global_result?.energy ?? null
}

export function extractValidation(job) {
  return unwrapPipelineOutput(job)?.validation || null
}

export function extractQubo(job) {
  return unwrapPipelineOutput(job)?.qubo_instance || null
}

export function extractConfig(job) {
  return unwrapJob(job)?.effective_config || job?.effective_config || null
}

export function assignmentRows(job) {
  const assignments = extractAssignments(job) || {}
  return Object.entries(assignments)
    .map(([entity, core]) => ({ entity, core }))
    .sort((a, b) => Number(a.entity) - Number(b.entity))
}

export function convergenceRows(job) {
  return extractConvergence(job).map((cost, index) => ({
    iteration: index + 1,
    cost,
  }))
}

export function mono(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'pending'
  return Number(value).toFixed(digits)
}
