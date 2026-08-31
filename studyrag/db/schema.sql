-- Run once against the Supabase database.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source file. File-level facts live here once instead of being repeated
-- across every chunk, which is what lets the dedupe rule be a plain UNIQUE constraint
-- rather than something fighting the chunk identity key.
CREATE TABLE IF NOT EXISTS documents (
    id           bigserial   PRIMARY KEY,
    course       text        NOT NULL,   -- module code, from the top-level directory
    doc_path     text        NOT NULL,   -- course-relative path, e.g. week4/lecture1.pdf
    content_hash text        NOT NULL,   -- SHA-256 of the raw file bytes
    doc_type     text        NOT NULL
                 CHECK (doc_type IN ('notes','slides','tutorial_q','tutorial_soln')),
    lecture      text,                   -- same for every chunk of the file, so it lives here
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    -- Identity: where the file sits. Renaming is an UPDATE of this row, not a re-ingest.
    UNIQUE (course, doc_path),
    -- Dedupe: the same bytes twice in one course is a copy, so skip it. No `ordinal`
    -- needed now that the fact is stored once. The same bytes in ANOTHER course are
    -- ingested independently — course isolation outranks storage savings.
    UNIQUE (course, content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial   PRIMARY KEY,
    document_id  bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    -- Denormalised on purpose. `course` is the retrieval filter and it must sit on the
    -- same row as the vector: a filtered ANN scan cannot reach through a join.
    course       text        NOT NULL,
    section      text,                   -- nearest sub-heading; varies per chunk
    ordinal      int         NOT NULL,   -- position within the document, 0-based
    page         int,                    -- printed slide number; not equal to ordinal
    content      text        NOT NULL,
    token_count  int         NOT NULL,
    embedding    vector(768) NOT NULL,
    embed_model  text        NOT NULL,   -- only knowable at insert time; never backfillable
    created_at   timestamptz NOT NULL DEFAULT now(),
    -- Set by the upsert on every write, so a row the last ingest refreshed is
    -- distinguishable from one it left behind as stale.
    updated_at   timestamptz NOT NULL DEFAULT now(),
    -- Identity. Makes re-ingest idempotent, and its index serves the parent join.
    UNIQUE (document_id, ordinal)
);

-- Deny-all by default. The backend connects as a BYPASSRLS role, so this only closes
-- the PostgREST/anon-key path — the browser never talks to Postgres directly.
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;

-- The course filter is on every single retrieval query, so it gets its own index.
-- UNVALIDATED at this size: with ~3k rows Postgres will likely seq-scan anyway.
-- Check pg_stat_user_indexes.idx_scan at Phase 4.
CREATE INDEX IF NOT EXISTS chunks_course_idx ON chunks (course);

-- Cosine distance, matching the normalised embeddings sentence-transformers emits.
-- Dimension must match settings.embed_dim; a mismatch fails loudly at insert, which is
-- the intent — embed_model on each row makes a half-re-embedded corpus detectable.
-- HNSW builds incrementally, so unlike ivfflat it is safe to create on an empty table
-- and never needs rebuilding as the corpus grows. Defaults (m=16, ef_construction=64)
-- are fine; tune recall at query time with `SET hnsw.ef_search`.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
