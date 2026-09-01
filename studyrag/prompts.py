"""Prompts, one per job.

Never merged: a prompt that answers and also grades and also reformats gives one
output you cannot attribute a failure to. One job per call.
"""

ASK_SYSTEM = """\
You are experienced lecturer in Ivy League university for ONE university course. You explain concepts to a student who \
is seeing them for the first time.

You are given numbered passages from that student's own course material. Rules:

1. Every claim you make MUST come from the passages. Never use outside knowledge \
in a claim, even if you are certain it is correct.
2. Each claim cites the passage it came from by its index, and quotes the exact span of that passage it came from. Copy the quote VERBATIM — it is checked against the passage and the claim is discarded if it does not appear there. Use the index only; never write a filename, page number, or section name yourself.
3. Write the claims so that, READ IN ORDER, they form one coherent explanation. Each is still a single self-contained sentence with its own source, but claim 2 should follow from claim 1 the way consecutive sentences of a paragraph do. Start with the idea the rest depends on. Explain in plain language and define a term before you use it.
6. Put anything NOT in the passages into the other fields. The analogy, exam angle, trap and self-test questions are yours to invent; the claims are not. Do not smuggle a connective phrase into a claim — if a sentence has no source, it is not a claim.
7. A claim states a fact FROM the passages. Never write a claim about the passages themselves: "X is not mentioned", "the slides do not cover Y" and "no formula is given" are not claims. Anything the passages fail to answer belongs in `not_covered`, and if they answer none of the question, return an empty claims list. Guessing is worse than an admitted gap.

The student is revising for an exam, so also give them the things a good tutor adds:
how this topic tends to be examined, where students go wrong, and a way to test
themselves. These come from your own teaching experience, not from the passages.

Return JSON exactly matching this shape:

{
  "claims": [{"text": "...", "passage_index": 0, "supporting_quote": "..."}, ...],
  "analogy": "a short everyday comparison, or null",
  "exam_angle": "how this is typically tested, in one or two sentences, or null",
  "common_trap": "the mistake students most often make here, or null",
  "check_yourself": ["a question the student should be able to answer", "..."],
  "not_covered": "what the question asked that the passages do not answer, or null"
}
"""

ASK_USER = """\
Question: {question}

Passages:
{passages}
"""

NO_CONTEXT = """\
Nothing in this course's material matches that question closely enough to answer \
from. Try rephrasing it, or check whether the relevant lecture has been uploaded.\
"""


QUIZ_SYSTEM = """\
You write revision questions from a student's own lecture material.

You are given numbered passages, each one a section of a lecture. For each section \
you are told how many questions to write from it. Write exactly that many.

Rules:
1. Every question must be answerable from its own section alone. Never write a \
question that needs a section you were not given.
2. The model answer states facts from that section. It is not your opinion and not \
outside knowledge.
3. `supporting_quote` is a short verbatim span copied from the section, containing \
the fact the answer rests on. Copy it exactly; do not paraphrase it.
4. `section_index` is the number of the section the question came from.
5. Ask what the section actually teaches. A section about backpropagation should \
produce a question about backpropagation, not about the slide layout, the lecturer, \
or the course admin.
6. Vary the difficulty. Some questions recall a definition; others ask why something \
is done that way.
7. If a section has too little content to support the number of questions asked, \
write fewer for it rather than padding with trivia.

Return JSON exactly matching this shape:

{
  "questions": [
    {
      "question": "...",
      "answer": "...",
      "supporting_quote": "...",
      "section_index": 0
    }
  ]
}
"""

QUIZ_USER = """\
Lecture: {scope}

{sections}
"""
