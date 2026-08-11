-- Run once against the Supabase database.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    doc_id       text        NOT NULL,   -- stable id for the source document
    doc_type     text        NOT NULL,   -- notes | slides | tutorial_q | tutorial_soln
    source_path  text        NOT NULL,
    lecture      text,                   -- top-level heading, e.g. "Lecture 4"
    section      text,                   -- nearest sub-heading
    ordinal      int         NOT NULL,   -- position within the document, 0-based
    content      text        NOT NULL,
    token_count  int         NOT NULL,
    embedding    vector(384) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, ordinal)
);

-- Deny-all by default. The backend connects as a BYPASSRLS role, so this only
-- closes the PostgREST/anon-key path — the browser never talks to Postgres directly.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

-- Quiz mode filters on these before ranking, so they need their own index.
CREATE INDEX IF NOT EXISTS chunks_meta_idx ON chunks (doc_type, lecture, section);

-- Cosine distance, matching the normalised embeddings sentence-transformers emits.
-- ivfflat needs rows present before it can build useful lists: create it AFTER the
-- first ingest, and re-create it when the corpus grows by an order of magnitude.
-- lists ~= sqrt(row_count) is the usual starting point.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
