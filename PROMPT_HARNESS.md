[ROLE & CONTEXT: RISKFORGE SYSTEM]
Repository: RiskForge (OIL SIH26165)
Target Environment: Native Linux x86_64, air-gapped CPU deployment, strict memory constraints.

You are implementing functionality strictly within the directory:
`{TARGET_DIRECTORY}`

Before writing any code:
1. Read `src/riskforge/core/contracts.py`. You are strictly forbidden from modifying this file.
2. Read `{TARGET_DIRECTORY}/CONTEXT.md`. Enforce all invariants, input/output schemas, and forbidden dependencies.
3. Check `docs/decisions/` for relevant architectural constraints.

Operational Rules:
- Return pure, functional production code for Linux execution.
- Respect subsystem boundaries. Do not import modules outside the declared matrix.
- If you believe an existing design choice in CONTEXT.md has flaws, you must write a formal ADR proposing the change instead of unilaterally breaking contracts.
