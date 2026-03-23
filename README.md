# Hybrid Quantum-Classical Testbed for OS Scheduling

A research testbed exploring hybrid quantum-classical approaches applied to process scheduling in operating systems. Uses QUBO (Quadratic Unconstrained Binary Optimization) formulations to model core assignment and time slot scheduling problems.

> **Status:** Work in progress — prototype stage.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- make

### Install `uv` (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart your terminal
```

---

## Getting Started

```bash
git clone <repo-url>
cd Hybrid-Quantum-Classical-Testbed-For-Operating-System-Scheduling
make install
```

---

## Commands

| Command | Description |
|---------|-------------|
| `make` | List all available commands |
| `make install` | Install all dependencies (`uv sync`) |
| `make run` | Run the project |
| `make add pkg=<name>` | Install a package and add it to `pyproject.toml` |
| `make remove pkg=<name>` | Remove a package from the project |
| `make freeze` | List installed packages |
| `make activate` | Print the command to manually activate the `.venv` |

### Examples

```bash
# Install all dependencies
make install

# Run the project
make run

# Add a new package
make add pkg=scipy

# Remove a package
make remove pkg=scipy

# Activate the .venv manually (to use python/pip directly)
make activate
# → copy and run: source .venv/bin/activate
```

---

## Project Structure

```
.
├── src/
│   ├── main.py
│   ├── qubo/
│   │   ├── qubo_solver.py          # Base solver (brute-force)
│   │   ├── qubo_core.py            # Core assignment solver
│   │   ├── qubo_time.py            # Time slot assignment solver
│   │   └── solver_checker.py       # Solution validation and global optimum comparison
│   └── tracer/
│       ├── tracer.py               # Base tracer class
│       └── specialized_tracers.py  # Memory and process tracers
├── pyproject.toml                  # Project dependencies (like package.json)
├── uv.lock                         # Lock file (like package-lock.json)
├── Makefile                        # Project commands
└── .venv/                          # Virtual environment (not versioned)
```

---

## Dependencies

| Package | Version |
|---------|---------|
| numpy   | ≥ 2.4.3 |
