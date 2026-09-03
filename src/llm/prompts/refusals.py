# The one line a prompt dictates that code reads back without knowing which
# prompt dictated it. A refusal is correct behaviour, reported apart from an
# answer — in the graders, the coverage counts, the greyed cells — and none
# of those readers know which prompt set the answer came from, so the two
# refusal lines have to be one list that neither prompt module can own.


def declines(answer: str) -> bool:
    """Whether an answer is the refusal its prompt allows instead of an
    answer — it opens with one of the mandated lines, NO NARRATIVE for a
    narrative and NO FRAMES for a frame analysis, puts forward no content
    about the query, and so is graded, aggregated and reported apart from an
    answer. A frames refusal may still carry its DISCARDED list: accounting
    for every tweet is asked of the writer either way, and a discard list
    reports no frame."""
    return answer.strip().upper().startswith(("NO NARRATIVE:", "NO FRAMES:"))
