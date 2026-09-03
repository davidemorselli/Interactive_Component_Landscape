# One summary's versions: the grading of each, the refinement loop that asks
# for the next, and which version a run stands on. Pure pipeline logic — no
# Flask, no SQL — shared by the runs that write versions (bulk_page,
# frames_page) and the views that read them back (task_views).

import math

from llm.pipeline.acueval import build_acueval_feedback
from llm.pipeline.rubric import build_rubric_feedback
from llm.prompts.refusals import declines

from .params import GRADERS, num, role_config
from .state import search_lock, speaker_names

# The refinement fields of one summary version, in their CSV column order.
# `warranted_grade` is filled for the words the writer declined to summarise and
# `rubric_grade` for the words it did: the two are never both present, and are
# kept apart because they answer different questions on incomparable scales.
# Every name here is a stored JSON key in the tasks table — never rename them.
FIELDS = ["summary", "rubric_grade", "rubric_comment", "acue_grade", "acue_comment",
          "warranted_grade", "warranted_comment", "budget_words"]

# The fields of FIELDS that hold a grade, rounded for the CSV.
GRADE_FIELDS = {"rubric_grade", "acue_grade", "warranted_grade"}


def acueval_claims(verified):
    """
    The claims ACUEval read out of a summary and how the tweets answered each,
    one per line. A claim the grader answered neither way is marked as such
    rather than dropped.
    """
    marks = {True: "correct", False: "wrong"}
    return "\n".join(f"- [{marks.get(supported, 'undecided')}] {unit}"
                     for unit, supported in verified["supported"].items())


def graded_version(word, summary, tweets, graders):
    """
    One summary version with both its grades and critiques.

    A narrative and a refusal are not graded on the same scale — the rubric
    for one, `warranted` for the other — and are reported in separate fields
    so that no mean, floor or colour ever puts the two together: a refusal
    correctly judged the only right answer is a 5 that says nothing about the
    quality of a narrative.
    """
    verified, score = graders["acueval"].evaluate_summary(summary, tweets)
    graded = graders["rubric"].evaluate_summary(word, summary, tweets)
    grade = None if graded is None else float(graded["grade"]["average"])
    # The grader's own reason for its grades, which it gives whatever it
    # graded — it goes with the grade, on whichever of the two scales the
    # version was graded.
    reason = "" if graded is None else str(graded["grade"].get("reason", ""))
    abstained = declines(summary)
    return {"summary": summary,
            "abstained": abstained,
            "acue_grade": None if math.isnan(score) else score,
            # As with the rubric below: the comment reported is what the grader
            # found, the critique the rewrite is driven by is kept aside.
            "acue_comment": acueval_claims(verified),
            "acue_feedback": build_acueval_feedback(verified),
            "rubric_grade": None if abstained else grade,
            "warranted_grade": grade if abstained else None,
            "rubric_comment": "" if abstained else reason,
            "warranted_comment": reason if abstained else "",
            # What the summary would be rewritten against is the rubric-shaped
            # critique, kept aside: it is empty at full marks, and that
            # emptiness is what stops the refinement.
            "rubric_feedback": "" if graded is None else build_rubric_feedback(graded),
            # How long the summary is, which is measured rather than graded and
            # so wears no colour and clears no floor. The critique it would be
            # rewritten against is empty unless it overran its budget.
            "budget_words": graders["budget"].evaluate_summary(summary),
            "budget_feedback": graders["budget"].feedback(summary)}


def rubric_grade_of(version):
    """
    The grade the rubric grader gave one version, on whichever of its two
    scales the version was graded — what the refinement loop compares against
    the rubric threshold, since a refusal it judged unwarranted is exactly as
    much a reason to write again as a narrative that fell short.
    """
    # .get: a word that produced nothing has a pseudo-version carrying only
    # the reason in place of its summary (task_views.versions_of), with no
    # grades to read.
    return (version.get("warranted_grade") if version.get("abstained")
            else version.get("rubric_grade"))


# Where an ungraded version ranks: below every grade one did give — it is not
# a good version, it is one nothing vouches for.
UNGRADED = -1.0


def version_rank(versions, i):
    """
    How good one version of a summary is, as a key its run's versions sort on.

    The order is the refinement loop's own (refine_versions): the rubric first,
    ACUEval next, and the run's own order last, so that of two versions nothing
    else tells apart the later wins — it is the one that answered the last
    critique, and the shorter one when the budget was what asked.
    """
    version = versions[i]
    rubric, acue = rubric_grade_of(version), version.get("acue_grade")
    return (UNGRADED if rubric is None else rubric,
            UNGRADED if acue is None else acue, i)


def selected_index(versions):
    """
    Which version of a summary the task stands on: the best graded of the run,
    not the last written. Refinement is not monotone — a rewrite answers one
    critique and can lose what the version before it had (an analysis graded
    4/5 came back 1/5 with an invented problem definition; another came back
    as the rewrite instructions themselves) — and every version was graded on
    the way past, so the grades are there to be read. A refusal ranks on the
    scale it was graded on (rubric_grade_of), the same one the loop weighs.
    """
    return max(range(len(versions)), key=lambda i: version_rank(versions, i))


def selected_version(versions):
    """The version of a summary the task stands on — the one selected_index
    picks out of the run."""
    return versions[selected_index(versions)]


def wants_refinement(refine, key, attempts, grade, feedback):
    """Whether one grader's settings ask for another rewrite of this version."""
    cfg = refine.get(key) or {}
    return bool(cfg.get("enabled") and attempts < num(refine.get("max_attempts"), 1, int)
                and grade is not None and grade < num(cfg.get("threshold"), 0)
                and feedback)


def wants_budget_refinement(refine, attempts, feedback):
    """
    Whether the word budget asks for another rewrite of this version. It has no
    threshold to compare against — a summary is over its budget or it is not,
    and the critique is empty unless it is — so this is `wants_refinement`
    without the grade.
    """
    cfg = refine.get("budget") or {}
    return bool(cfg.get("enabled") and attempts < num(refine.get("max_attempts"), 1, int)
                and feedback)


def refine_versions(versions, word, first, tweets, graders, refine):
    """
    Fill `versions` with every version of one summary: the first, then a
    rewrite per round of critique, while the refinement settings ask for one.
    It appends as it goes, so the versions already graded survive a call that
    fails midway.
    """
    versions.append(graded_version(word, first, tweets, graders))
    attempts = {"rubric": 0, "acueval": 0, "budget": 0}
    while True:
        cur = versions[-1]
        # The rubric comes first: a round it triggers never also refines
        # against the ACUEval critique. The budget comes last of the three —
        # what a summary says is worth more than how long it is, and shortening
        # one the graders are still unhappy with is work thrown away.
        if wants_refinement(refine, "rubric", attempts["rubric"],
                            rubric_grade_of(cur), cur["rubric_feedback"]):
            key, agent, feedback = "rubric", graders["rubric"], cur["rubric_feedback"]
        elif wants_refinement(refine, "acueval", attempts["acueval"],
                              cur["acue_grade"], cur["acue_feedback"]):
            key, agent, feedback = "acueval", graders["acueval"], cur["acue_feedback"]
        elif wants_budget_refinement(refine, attempts["budget"],
                                     cur.get("budget_feedback", "")):
            key, agent, feedback = "budget", graders["budget"], cur["budget_feedback"]
        else:
            return
        attempts[key] += 1
        revised = agent.refine_summary(word, tweets, cur["summary"], feedback)
        if revised is None:  # the rewrite lost the mandatory opening
            return
        # Which critique sent the version before this one back. The versions
        # were always kept; what drove each rewrite was not, and the view has
        # nothing to say about a task run before this line.
        versions.append(graded_version(word, revised, tweets, graders)
                        | {"refined_for": key})


def task_speakers(p):
    """
    Bare speaker names to restrict a task's retrieval to, per phase of its
    words, or None to scan every speaker. Resolved once, before the fan-out —
    the component cache behind the lookups is not thread-safe — and raising
    when the restriction leaves no speaker at all, which would otherwise be
    reported word by word as retrieval failures.
    """
    source = p.get("speakers", "all")
    if source == "all":
        return None
    component = int(p["component"])
    n_extreme = num(p.get("n_extreme"), 50, int)
    phases = {str(item["phase"]) for item in p.get("words") or []}
    with search_lock:
        names = {phase: speaker_names(source, component, n_extreme, phase)
                 for phase in phases}
    if not any(names.values()):
        raise ValueError(f"component {component} has no {source}")
    return names


def build_agents(p, level):
    """The (writer, graders) of one level of a run, from the task's model
    boxes. Both levels of a set are built from the same boxes: what the merge
    needs of a model is what a word's answer needs of it, and the page asked
    twice for one answer.

    `level` is what actually differs between the levels — the word level reads
    tweets and the merge level reads the answers written from them — and each
    passes its own (llm.pipeline.levels).
    """
    return (level.writer(**role_config(p, "summary")),
            {role: level.agent(role, **role_config(p, role)) for role in GRADERS})
