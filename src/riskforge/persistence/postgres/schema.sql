-- RiskForge PostgreSQL Schema v1
-- Idempotent DDL: safe to re-apply.

CREATE TABLE IF NOT EXISTS incident_results (
    log_id           TEXT        PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL,
    asset_id         TEXT        NOT NULL,
    asset_type       TEXT        NOT NULL,
    routing          TEXT        NOT NULL,
    calibrated_sif_p_score DOUBLE PRECISION NOT NULL,
    record_json      JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_incident_results_timestamp_log_id
    ON incident_results (timestamp, log_id);
CREATE INDEX IF NOT EXISTS idx_incident_results_asset_id
    ON incident_results (asset_id);
CREATE INDEX IF NOT EXISTS idx_incident_results_asset_type
    ON incident_results (asset_type);
CREATE INDEX IF NOT EXISTS idx_incident_results_routing
    ON incident_results (routing);
CREATE INDEX IF NOT EXISTS idx_incident_results_score
    ON incident_results (calibrated_sif_p_score);

CREATE TABLE IF NOT EXISTS review_decisions (
    decision_id  TEXT           PRIMARY KEY,
    log_id       TEXT           NOT NULL,
    action       TEXT           NOT NULL,
    reviewer_id  TEXT           NOT NULL,
    decided_at   TIMESTAMPTZ    NOT NULL,
    reason       TEXT           NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_decisions_incident
    ON review_decisions (log_id, decided_at, decision_id);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id    TEXT           PRIMARY KEY,
    log_id      TEXT           NOT NULL,
    event_type  TEXT           NOT NULL,
    actor_id    TEXT           NOT NULL,
    occurred_at TIMESTAMPTZ    NOT NULL,
    reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_events_incident
    ON audit_events (log_id, occurred_at, event_id);
