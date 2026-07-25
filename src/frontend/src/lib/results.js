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

export function extractDurationMs(job) {
  const wrapped = unwrapJob(job)
  const output = unwrapPipelineOutput(job)
  return wrapped?.duration_ms ?? output?.total_solve_time_ms ?? output?.result?.solve_time_ms ?? null
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

export function extractPipelineLabel(job) {
  const outputType = unwrapJob(job)?.output_type
  if (outputType === 'IterativeSchedulingOutput') return 'iterative'
  if (outputType === 'SchedulingOutput') return 'direct'
  return 'pending'
}

export function extractProcesses(job) {
  const output = unwrapPipelineOutput(job)
  const config = extractConfig(job)
  const snapshotProcesses = output?.used_snapshot?.processes
  const workloadEntities = output?.used_workload?.entities

  const rawProcesses = Array.isArray(snapshotProcesses) && snapshotProcesses.length
    ? snapshotProcesses.map((process) => ({
      entity: String(process.pid),
      weight: Number(process.cpu_weight),
      sourceLabel: process.command,
    }))
    : Array.isArray(workloadEntities) && workloadEntities.length
      ? workloadEntities.map((entity) => ({
        entity: String(entity.entity_id),
        weight: Number(entity.cpu_weight),
        sourceLabel: entity.label,
      }))
      : Array.isArray(config?.weights)
        ? config.weights.map((weight, index) => ({
          entity: String(1000 + index),
          weight: Number(weight),
          sourceLabel: `p_${index}`,
        }))
        : []

  return rawProcesses
    .sort((a, b) => Number(a.entity) - Number(b.entity))
    .map((process, index) => ({
      ...process,
      index,
      label: `p_${index}`,
    }))
}

export function assignmentRows(job) {
  return assignmentRowsFor(job, extractAssignments(job) || {})
}

export function assignmentRowsFor(job, assignments = {}) {
  const processByEntity = new Map(extractProcesses(job).map((process) => [process.entity, process]))
  return Object.entries(assignments)
    .map(([entity, core]) => {
      const process = processByEntity.get(String(entity))
      return {
        entity,
        core: Number(core),
        label: process?.label ?? `p_${Number(entity) >= 1000 ? Number(entity) - 1000 : entity}`,
        processIndex: process?.index ?? Number(entity),
        weight: process?.weight ?? 0,
      }
    })
    .filter((row) => Number.isFinite(row.core))
    .sort((a, b) => a.processIndex - b.processIndex)
}

export function coreAssignmentGroups(job) {
  return coreAssignmentGroupsFor(job, extractAssignments(job) || {})
}

export function coreAssignmentGroupsFor(job, assignments = {}) {
  const config = extractConfig(job)
  const qubo = extractQubo(job)
  const rows = assignmentRowsFor(job, assignments)
  const declaredCoreCount = Number(config?.num_cores ?? qubo?.num_cores)
  const inferredCoreCount = Math.max(0, ...rows.map((row) => row.core)) + 1
  const coreCount = Number.isFinite(declaredCoreCount) && declaredCoreCount > 0
    ? declaredCoreCount
    : inferredCoreCount
  const groups = Array.from({ length: Math.max(coreCount, 0) }, (_, core) => ({
    core,
    load: 0,
    processes: [],
  }))

  rows.forEach((row) => {
    if (!groups[row.core]) {
      groups[row.core] = { core: row.core, load: 0, processes: [] }
    }
    groups[row.core].processes.push(row)
    groups[row.core].load += Number(row.weight || 0)
  })

  const totalLoad = groups.reduce((sum, group) => sum + group.load, 0)
  const targetLoad = groups.length ? totalLoad / groups.length : 0
  const maxLoad = Math.max(targetLoad, ...groups.map((group) => group.load), 0)

  return groups.map((group) => ({
    ...group,
    loadPct: maxLoad ? (group.load / maxLoad) * 100 : 0,
    targetLoad,
    imbalanceFromTarget: group.load - targetLoad,
  }))
}

export function extractClassicalAssignments(job) {
  const assignments = extractValidation(job)?.global_assignments
  return assignments && Object.keys(assignments).length ? assignments : null
}

export function coreLoadComparisonRows(job) {
  const currentGroups = coreAssignmentGroups(job)
  const classicalAssignments = extractClassicalAssignments(job)
  const classicalGroups = classicalAssignments ? coreAssignmentGroupsFor(job, classicalAssignments) : []

  return currentGroups.map((group, index) => ({
    core: `core ${group.core}`,
    current: group.load,
    classical: classicalAssignments ? classicalGroups[index]?.load ?? null : null,
    average: group.targetLoad,
  }))
}

export function coreLoadRows(job) {
  return coreAssignmentGroups(job).map((group) => ({
    core: `core ${group.core}`,
    load: group.load,
    target: group.targetLoad,
  }))
}

export function solverRunRows(job) {
  const output = unwrapPipelineOutput(job)
  const runs = Array.isArray(output?.solver_results) && output.solver_results.length
    ? output.solver_results
    : output?.result
      ? [output.result]
      : []

  return runs.map((run, index) => ({
    stage: runs.length > 1 ? `sub ${index + 1}` : 'direct',
    energy: Number(run.energy),
    solveMs: Number(run.solve_time_ms ?? 0),
    feasible: run.is_feasible,
  }))
}

function balanceStats(groups) {
  const loads = groups.map((group) => Number(group.load || 0))
  if (!loads.length) {
    return {
      average: null,
      max: null,
      min: null,
      imbalance: null,
      normalizedImbalance: null,
    }
  }

  const total = loads.reduce((sum, load) => sum + load, 0)
  const average = total / loads.length
  const max = Math.max(...loads)
  const min = Math.min(...loads)
  const imbalance = max - min

  return {
    average,
    max,
    min,
    imbalance,
    normalizedImbalance: average ? imbalance / average : 0,
  }
}

export function balanceReferenceSummary(job) {
  const current = balanceStats(coreAssignmentGroups(job))
  const classicalAssignments = extractClassicalAssignments(job)
  const classical = classicalAssignments ? balanceStats(coreAssignmentGroupsFor(job, classicalAssignments)) : null

  return {
    hasClassical: Boolean(classical),
    current,
    classical,
    average: current.average,
  }
}

export function resultSummary(job) {
  const output = unwrapPipelineOutput(job)
  const validation = extractValidation(job)
  const reference = balanceReferenceSummary(job)
  const loadImbalance = output?.load_imbalance ?? reference.current.imbalance
  return {
    pipeline: extractPipelineLabel(job),
    feasible: validation?.valid ?? output?.result?.is_feasible ?? output?.global_result?.is_feasible ?? null,
    optimal: validation?.is_optimal ?? null,
    optimalityGap: output?.alpha ?? null,
    energy: extractEnergy(job),
    globalEnergy: validation?.global_energy ?? null,
    variables: extractQubo(job)?.num_variables ?? null,
    subQubos: output?.num_sub_qubos ?? solverRunRows(job).length,
    durationMs: extractDurationMs(job),
    loadImbalance,
    normalizedLoadImbalance: reference.current.normalizedImbalance,
    averageCoreLoad: reference.current.average,
  }
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

export function compactNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'pending'
  const numeric = Number(value)
  if (numeric === 0) return '0'
  if (Math.abs(numeric) >= 1000 || Math.abs(numeric) < 0.001) return numeric.toExponential(2)
  return Number(numeric.toFixed(digits)).toString()
}

export function statusText(value) {
  if (value === true) return 'yes'
  if (value === false) return 'no'
  return 'pending'
}

export function optimalityText(summary) {
  if (summary.feasible === false) return 'The decoded assignment failed validation.'
  if (summary.optimal === true) return 'The decoded assignment matched the brute-force optimum for this run.'
  if (summary.optimal === false) return 'The run produced a feasible assignment, but it did not match the brute-force optimum.'
  if (summary.feasible === true) return 'The decoded assignment is feasible. No certified optimum was returned for this run.'
  return 'Run validation has not completed yet.'
}
