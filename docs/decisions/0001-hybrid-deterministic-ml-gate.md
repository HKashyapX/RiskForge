# ADR-0001: Hybrid Deterministic Energy-Barrier Gate and ML Output

## Status
Accepted

## Context
Standard ML classifiers fail statutory audits under OISD-STD-105 when high-energy breaches are misclassified as benign operational noise due to distribution shifts or uncalibrated sigmoid tails.

## Decision
Enforce a physical deterministic override gate in parallel with neural inference:
If critical energy is present ($E \ge E_{crit}$), critical primary barrier has failed ($B_{crit} \in \{Failed, Missing\}$), and worker exposure vector exists ($X_{worker} = 1$), the incident is forced to SIF-P ($score \ge 0.95$) regardless of model logits.

## Consequences
- Zero false negatives on explicit statutory hazard violations.
- Neural model handles ambiguous, multi-label, and narrative contextual nuance.
