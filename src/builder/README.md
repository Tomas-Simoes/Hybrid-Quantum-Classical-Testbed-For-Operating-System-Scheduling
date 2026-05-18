# builder

This module translates scheduling data into QUBO matrices.

## CoreAssignmentBuilder

`builder_core.py` builds the main process-to-core assignment QUBO.

For `N` entities and `K` cores, each binary variable `x_{i,j}` means:

```text
entity i is assigned to core j
```

The QUBO has `N * K` variables, and `variable_map` maps each flat variable index
to `(entity_id, core_id)`.

Important terms:

- `P`: penalty weight for one-hot assignment constraints.
- `L_avg`: target average core load, `sum(weights) / K`.
- `QUBOInstance.Q`: the matrix optimized by solvers.

The builder encodes both load balancing and one-hot assignment constraints.
Feasibility is later decoded by checking that every entity has exactly one active
core variable.

## TimeAssignmentBuilder

`builder_time.py` is an experimental time-slot builder. It uses a prior
core-assignment result and creates a QUBO over time slots. This is not the main
runtime path right now.

