# RFC-[NUMBER]: [Proposed Contract Mutation]

## Author
Developer: [Name]
Model Used: [e.g., Claude 3.5 Sonnet / GPT-4o]
Target Subsystem: [e.g., src/riskforge/modeling]

## 1. Proposed Schema Mutation
```diff
--- a/src/riskforge/core/contracts.py
+++ b/src/riskforge/core/contracts.py
@@ ... @@
```

## 2. Problem Statement & Failure Mode
[Explain why current contracts.py schema cannot support the required functionality]

## 3. Impact Analysis Across Subsystems
- normalization/: [Impact / No Impact]
- supervision/: [Impact / No Impact]
- modeling/: [Impact / No Impact]
- serving/: [Impact / No Impact]
- metrics/: [Impact / No Impact]

## 4. Alternative Workarounds Evaluated & Rejected
[Explain why this cannot be solved inside your own subsystem without modifying the global contract]
