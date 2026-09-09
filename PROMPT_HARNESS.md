[ROLE & CONTEXT: RISKFORGE SYSTEM]
Repository: RiskForge (OIL SIH26165)
Target Environment: Cross-platform Python deployment with CPU execution, support for Windows, Linux, and macOS, and deployment environments ranging from local development to controlled/offline installations.

You are implementing functionality strictly within the directory:
`{TARGET_DIRECTORY}`

Before writing any code:
1. Read `src/riskforge/core/contracts.py`. You are strictly forbidden from modifying this file.
2. Read `{TARGET_DIRECTORY}/CONTEXT.md`. Enforce all invariants, input/output schemas, and forbidden dependencies.
3. Check `docs/decisions/` for relevant architectural constraints.

Operational Rules:
- Return portable, functional production code that is not dependent on a specific operating system, shell, filesystem layout, CPU architecture, or desktop environment.
- Respect subsystem boundaries. Do not import modules outside the declared matrix.
- Use cross-platform Python and standard-library facilities where practical; do not assume POSIX-only behavior, shell commands, path syntax, process semantics, or OS-specific services.
- Keep environment-specific deployment settings in configuration and deployment documentation rather than embedding them in application code or general engineering prompts.
- If you believe an existing design choice in CONTEXT.md has flaws, you must write a formal ADR proposing the change instead of unilaterally breaking contracts.
