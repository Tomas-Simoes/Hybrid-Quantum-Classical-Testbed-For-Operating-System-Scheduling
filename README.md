# Hybrid Quantum-Classical Testbed for OS Scheduling

A research prototype for studying hybrid quantum-classical process scheduling.

The current main problem is **process-to-core assignment**: given a workload of
process CPU weights and a target number of cores, the project builds a QUBO,
solves it with QAOA through PennyLane, and compares small instances against an
exact brute-force oracle.

> Status: work in progress. The code is useful for experiments, diagnostics, and
> visual exploration, but it is still a prototype rather than a production
> scheduler.

## What It Does

The default scheduling flow is:

```text
SystemSnapshot or preset weights
        |
        v
CoreAssignmentBuilder -> QUBOInstance
        |
        +--> PennylaneSolver  -> QAOA candidate
        |
        +--> BruteForceSolver -> exact reference, small QUBOs only
        |
        v
SolverValidator -> feasibility, optimality, optimality gap
```

For larger workloads, the engine switches to an iterative sub-QUBO pipeline:

```text
Global QUBO -> heuristic partition -> sub-QUBO solves -> bias propagation
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- make

Install `uv` once per machine:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## Getting Started

```bash
git clone github.com/Tomas-Simoes/Hybrid-Quantum-Classical-Testbed-For-Operating-System-Scheduling
cd Hybrid-Quantum-Classical-Testbed-For-Operating-System-Scheduling
make install
```

Run the CLI:

```bash
make run
```

## Commands

| Command | Description |
| --- | --- |
| `make` | List available commands |
| `make install` | Install dependencies with `uv sync` |
| `make run` | Run `src/core/main.py` |
| `make add pkg=<name>` | Add a dependency |
| `make remove pkg=<name>` | Remove a dependency |
| `make freeze` | List installed packages |
| `make activate` | Print the command to activate the local virtualenv |
| `make spy` | Record a py-spy profile to `profile.svg` |

## Module Docs

Detailed notes live next to the code they describe:

- [core](src/core/README.md): engine, contracts, UI entry points, and project map
- [builder](src/core/builder/README.md): QUBO builders
- [solver](src/core/solver/README.md): QAOA, brute force, validation, metrics
- [pipeline](src/core/pipeline/README.md): default and iterative execution flows
- [decomposition](src/core/decomposition/README.md): clustering and sub-QUBO partitioning
- [tracer](src/core/tracer/README.md): live process snapshot collection
- [visualizer](src/core/visualizer/README.md): plotting surfaces
- [abstract](src/core/abstract/README.md): base interfaces
