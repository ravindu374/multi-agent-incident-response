-- Warehouse bootstrap. Runs once, on an empty data volume.
-- dbt owns the staging and marts schemas; the pipeline writes raw.* and ops.job_runs.

CREATE SCHEMA raw;
CREATE SCHEMA ops;

-- Service and people roles --------------------------------------------------
CREATE ROLE etl           LOGIN PASSWORD 'etl_pw';          -- ingest job
CREATE ROLE transformer   LOGIN PASSWORD 'transformer_pw';  -- dbt
CREATE ROLE dashboard     LOGIN PASSWORD 'dashboard_pw';    -- metrics-api
CREATE ROLE app_reporting LOGIN PASSWORD 'reporting_pw';    -- nightly reporting service
CREATE ROLE analyst       LOGIN PASSWORD 'analyst_pw';      -- humans with notebooks
CREATE ROLE maint         LOGIN PASSWORD 'maint_pw';        -- ad-hoc maintenance scripts
CREATE ROLE agent_ro      LOGIN PASSWORD 'agent_ro_pw';     -- read-only role for agent tools (later phases)

-- Session limits. These turn slowness into the timeouts the pipeline reports.
ALTER ROLE etl         SET lock_timeout = '3s';
ALTER ROLE etl         SET statement_timeout = '10s';
ALTER ROLE transformer SET lock_timeout = '5s';
ALTER ROLE transformer SET statement_timeout = '15s';
ALTER ROLE dashboard   SET statement_timeout = '5s';

-- Raw layer -----------------------------------------------------------------
CREATE SEQUENCE raw.batch_id_seq;

CREATE TABLE raw.orders (
    order_id    bigint      NOT NULL,
    created_at  timestamptz NOT NULL,
    customer_id integer,
    store_id    integer,
    amount      numeric(10, 2),
    status      text,
    batch_id    bigint      NOT NULL,
    loaded_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX orders_order_id_idx ON raw.orders (order_id);
CREATE INDEX orders_batch_id_idx ON raw.orders (batch_id);

CREATE TABLE raw.orders_quarantine (LIKE raw.orders INCLUDING DEFAULTS);
ALTER TABLE raw.orders_quarantine
    ADD COLUMN quarantined_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN reason text;

CREATE TABLE raw.ingest_state (
    source        text PRIMARY KEY,
    last_order_id bigint      NOT NULL DEFAULT 0,
    updated_at    timestamptz NOT NULL DEFAULT now()
);
INSERT INTO raw.ingest_state (source) VALUES ('orders_api');

-- Job metadata (what an orchestrator UI would show) -------------------------
CREATE TABLE ops.job_runs (
    run_id         bigserial PRIMARY KEY,
    job            text        NOT NULL,
    started_at     timestamptz NOT NULL,
    finished_at    timestamptz NOT NULL DEFAULT now(),
    status         text        NOT NULL CHECK (status IN ('success', 'failed')),
    rows_processed integer,
    error          text,
    details        jsonb
);
CREATE INDEX job_runs_job_finished_idx ON ops.job_runs (job, finished_at DESC);

-- dbt-managed schemas ---------------------------------------------------------
CREATE SCHEMA staging AUTHORIZATION transformer;
CREATE SCHEMA marts AUTHORIZATION transformer;

-- Grants --------------------------------------------------------------------
GRANT USAGE ON SCHEMA raw, ops TO etl, transformer;
GRANT SELECT, INSERT ON raw.orders TO etl;
GRANT SELECT, UPDATE ON raw.ingest_state TO etl;
GRANT USAGE ON SEQUENCE raw.batch_id_seq TO etl;
GRANT INSERT ON ops.job_runs TO etl, transformer;
GRANT USAGE ON SEQUENCE ops.job_runs_run_id_seq TO etl, transformer;

GRANT SELECT ON raw.orders TO transformer;
GRANT CREATE ON DATABASE warehouse TO transformer;

GRANT USAGE ON SCHEMA marts TO dashboard;
ALTER DEFAULT PRIVILEGES FOR ROLE transformer IN SCHEMA marts GRANT SELECT ON TABLES TO dashboard;

GRANT USAGE ON SCHEMA raw TO maint;
GRANT SELECT, DELETE ON raw.orders TO maint;

GRANT pg_read_all_data TO app_reporting, analyst, agent_ro;
GRANT pg_monitor TO agent_ro;
