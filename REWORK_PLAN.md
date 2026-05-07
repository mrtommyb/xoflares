# xoflares rework plan

This document defines a concrete path for the repository rework requested in the issue titled "Complete rework of this repo":
- adopt a flare model based on Lupita Tovar's work (currently available in notebooks),
- replace PyMC-based sampling with an alternative that still integrates with transit modeling workflows.

## Scope

1. Rework flare modeling implementation into a clean, testable package API.
2. Decouple sampling from PyMC3/Theano-era dependencies.
3. Preserve interoperability with transit model tools used by the exoplanet workflow.

## Phase 1 — discovery and baseline

- Locate and document the notebook implementation that reproduces Lupita Tovar's flare model.
- Extract current public API behavior from `xoflares/xoflares.py` and tests.
- Define benchmark datasets (simulated + real-star examples already in `notebooks/`).
- Freeze current outputs for regression comparisons (light curves + integrals).

Deliverables:
- `docs/model-notes.md` (source equations, assumptions, parameter definitions).
- baseline regression fixtures for current model behavior.

## Phase 2 — model extraction and architecture cleanup

- Implement a dedicated flare model module with:
  - pure NumPy reference implementation,
  - optional accelerated backend interface (pluggable).
- Add clear parameter validation (shape checks, positive FWHM, finite values).
- Keep existing high-level helper functions available via compatibility wrappers.

Deliverables:
- new flare model API with tests comparing against notebook outputs.
- deprecation notes for legacy internals if function names/paths change.

## Phase 3 — sampler migration away from PyMC

- Define sampler requirements:
  - posterior inference for flare parameters,
  - support for priors used in existing notebooks,
  - compatibility with transit model likelihood composition.
- Evaluate replacement options:
  - NumPyro/JAX (NUTS/HMC; strong modern probabilistic stack),
  - BlackJAX with custom model wrappers,
  - deterministic optimization + bootstrap fallback for lightweight use cases.
- Select one primary path and one fallback path.

Recommendation for initial implementation:
- Primary: NumPyro-based inference backend.
- Fallback: SciPy optimization path for non-probabilistic runs.

Deliverables:
- backend abstraction (`samplers/` with common interface).
- one working non-PyMC probabilistic backend integrated into example workflow.

## Phase 4 — transit interoperability

- Define a shared log-likelihood interface so flare + transit components compose cleanly.
- Add integration examples combining flare and transit components in a single fit.
- Validate parity against current notebook-level workflows.

Deliverables:
- integration tests for flare+transit joint modeling.
- migrated example notebook(s) using the new backend.

## Phase 5 — release hardening

- Update packaging/dependencies to remove PyMC3/Theano-era requirements.
- Expand CI to run tests on supported Python versions and dependency sets.
- Publish migration guide for downstream users.

Deliverables:
- updated `requirements.txt`/packaging metadata.
- migration guide and release notes.

## Proposed near-term milestones

1. **Milestone A (1–2 weeks):** finalize model notes and extraction plan from notebooks.
2. **Milestone B (2–3 weeks):** land new flare model module + regression tests.
3. **Milestone C (2–4 weeks):** integrate new sampler backend and transit interoperability example.

## Definition of done

- Flare model implementation is based on the identified Lupita Tovar notebook formulation and is unit-tested.
- No required sampling path depends on PyMC3.
- At least one maintained workflow demonstrates flare + transit interoperability end-to-end.
