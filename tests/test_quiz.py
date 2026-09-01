"""Allocation is pure arithmetic over bucket sizes, so it is tested without a database."""

from studyrag.modes.quiz import Bucket, allocate, plan_buckets


def bucket(n_chunks: int, label: str = "section", first: int = 0) -> Bucket:
    return Bucket(
        document_id=1, doc_path="deck.pdf", lecture="Lecture 4", label=label,
        first_ordinal=first, last_ordinal=first + n_chunks - 1, n_chunks=n_chunks,
    )


def test_every_bucket_gets_one_when_questions_allow():
    counts = allocate([bucket(n) for n in (15, 4, 2)], 3)
    assert counts == [1, 1, 1]


def test_spare_questions_follow_bucket_size():
    counts = allocate([bucket(n) for n in (15, 5)], 6)
    assert sum(counts) == 6
    assert counts[0] > counts[1]


def test_biggest_buckets_win_when_questions_run_out():
    counts = allocate([bucket(n) for n in (15, 13, 1, 1)], 2)
    assert counts == [1, 1, 0, 0]


def test_allocation_always_sums_to_the_request():
    buckets = [bucket(n) for n in (15, 13, 12, 10, 7, 6, 6, 5, 4, 2, 2, 1, 1)]
    for n in range(1, 26):
        assert sum(allocate(buckets, n)) == n


def test_no_buckets_allocates_nothing():
    assert allocate([], 5) == []


def test_unsectioned_run_is_split_by_reading_order():
    planned = plan_buckets([bucket(20, label="slides 1-20")])
    assert len(planned) == 3
    assert [b.n_chunks for b in planned] == [8, 8, 4]
    assert planned[0].label == "slides 1-8"


def test_small_sections_are_left_alone():
    # Merging them would make the citation name a section the fact did not come from.
    planned = plan_buckets([bucket(9, "Backpropagation"), bucket(1, "Final code", first=9)])
    assert [b.label for b in planned] == ["Backpropagation", "Final code"]


def test_bucket_key_survives_a_label_collision():
    # Two documents in one lecture each open with an unsectioned slide, so both are
    # labelled "slides 1-1". Coverage counts identity, not the display label.
    first = bucket(1, label="slides 1-1")
    second = bucket(1, label="slides 1-1").model_copy(update={"document_id": 2})
    assert first.label == second.label
    assert first.key != second.key
