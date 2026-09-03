# How the pages talk to the app: every parameter they send with its parser and
# default, the model boxes, and the small answer helpers the routes share.

import functools

from flask import Response, jsonify

from .app import BadInput, app
from .state import covered_words, landscape

# What the model boxes start on, by role. The boxes are free text — any
# identifier OpenRouter knows is valid — and the "browse models ↗" link beside
# each section is where to find one.
DEFAULT_SUMMARY_MODEL = "openai/gpt-5.6-luna"
DEFAULT_ACUEVAL_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_GRADER_MODEL = "openai/gpt-5.6-luna"


# The phases a drawn view can span: the four of the corpus, all four at once,
# or every phase pooled into one.
PHASES = ("1", "2", "3", "4", "all", "pooled")

# What a run page offers instead: 'all' is a way of looking — four panels,
# each computed on its own — not a unit of work, and a run under it quietly
# meant four runs wearing one name. So the choice is one single span, with
# 'pooled' as the whole corpus.
RUN_PHASES = tuple(p for p in PHASES if p != "all")


def flag(value):
    """A checkbox as the pages send it."""
    return value == "1"


def rounded(value):
    """round() that lets a None through — for optional numbers going to a page."""
    return None if value is None else round(value, 2)


def phase_filter(phase):
    """The corpus phase filter: the phase's own number, or None when the page
    asks for every phase."""
    return int(phase) if phase in ("1", "2", "3", "4") else None


# The component the pages open on, when the decomposition kept it; a run whose
# components came out differently falls back to the first one retained.
DEFAULT_COMPONENT = (201 if 201 in landscape.selected_components
                     else landscape.selected_components[0])

# Each entry is (parser, default, tags). The tags say which endpoints read the
# parameter, and the name lists below are derived from them — so a parameter
# cannot be wired into one list and forgotten in another:
#   nearest      the centroid and nearest-word search (/render and /words)
#   floor        the three candidate-word floors, a value behind a checkbox each
#   grouping     the found-word clustering — /words and the landscape lists
#                (kept out of `floor`: /export.csv turns floor names into columns)
#   word_source  the run pages' word-source choice and its family settings —
#                the same names the landscape page draws by, so both share
#                their defaults and the Landscape.component cache
#   drawn        what only /render draws
#   label_floor  the two label-family frequency floors, folded by drawn_kwargs
# The defaults are also what every page's controls start on (page_context
# hands them to the markup as `d`).
PARAMS = {
    # The centroid and the nearest-word search, which /render and /words share.
    "component":           (int, DEFAULT_COMPONENT, "nearest"),
    "phase":               (str, "pooled", "nearest"),
    "unit_norm":           (flag, True, "nearest"),
    "mean_centre":         (flag, True, "nearest"),
    "centroid_source":     (str, "extreme speakers", "nearest"),
    "n_extreme":           (int, 50, "nearest"),
    "word_filter":         (str, "extreme words", "nearest"),
    "k_extreme":           (int, 200, "nearest"),
    "topn":                (int, 50, "nearest"),
    # Three floors on the candidate words, each a value behind its own checkbox.
    "min_count_filter":    (flag, True, "floor"),
    "min_count":           (int, 10, "floor"),
    "similarity_filter":   (flag, True, "floor"),
    "min_similarity":      (float, 0.5, "floor"),
    "tweet_filter":        (flag, False, "floor"),
    "word_min_tweets":     (int, 5, "floor"),
    "word_min_similarity": (float, 0.3, "floor"),
    # Grouping of the found words into clusters of similar ones: the run pages
    # turn each cluster into one retrieval query; the landscape page only folds
    # its displayed lists — the plot and the export keep every word on its own.
    "group_words":         (flag, True, "grouping"),
    "group_similarity":    (float, 0.5, "grouping"),
    # Where the run pages' word list comes from: the centroid search above, or
    # the component's own strong/extreme words as the landscape page labels
    # them. The landscape page never sends it and stays on the search.
    "word_source":         (str, "centroid", "word_source"),
    # What only /render draws — the label families' count, floor and order also
    # feed the run pages' word source, under the same names. The tweet floor
    # and the grouping above apply to whichever source is chosen: one set of
    # controls, not one per family.
    "show_strong_speakers":  (flag, False, "drawn"),
    "strong_labels":         (flag, False, "drawn"),
    "show_extreme_speakers": (flag, False, "drawn"),
    "extreme_labels":        (flag, False, "drawn"),
    "n_extreme_speakers":  (int, 50, "drawn"),
    "n_extreme_words":     (int, 10, "drawn word_source"),
    "n_strong_words":      (int, 20, "drawn word_source"),
    "strong_min_count_filter": (flag, False, "label_floor word_source"),
    "strong_min_count":    (int, 10, "label_floor word_source"),
    "strong_order":        (str, "corpus frequency", "drawn word_source"),
    "extreme_min_count_filter": (flag, False, "label_floor word_source"),
    "extreme_min_count":   (int, 10, "label_floor word_source"),
    "extreme_order":       (str, "component strength", "drawn word_source"),
    "spread":              (float, 0.25, "drawn"),
    "scale":               (float, 1.0, "drawn"),
}


def named(*groups):
    """The parameters carrying any of these tags, in table order — which for
    the floor names is also /export.csv's column order."""
    return [name for name, (_, _, tags) in PARAMS.items()
            if any(g in tags.split() for g in groups)]


# The arguments Landscape.nearest takes, and the ones only render() takes.
NEAREST = named("nearest")
DRAWN = named("drawn")

# What each page collects from its controls: /bulk asks for words, / draws them.
WORD_PARAMS = named("nearest", "floor")
RENDER_PARAMS = named("nearest", "floor", "grouping", "drawn", "label_floor")
RUN_WORD_PARAMS = named("nearest", "floor", "grouping", "word_source")


def read(args, name):
    """
    One parameter as the page sent it, or its fallback when it did not — and
    equally when what it sent cannot be read as the parameter's type: a number
    box the user cleared (or typed a letter into) sends '', the ordinary state
    of a half-edited form.
    """
    parse, default, _ = PARAMS[name]
    raw = args.get(name)
    if raw is None:
        return default
    try:
        value = parse(raw)
    except (TypeError, ValueError):
        return default
    # These two name what is looked up rather than how much of it: a component
    # outside the retained ones indexes the ICA matrix anyway (-1 answers for
    # the last component, plausibly and wrongly), and a phase outside the six
    # reaches lookups with no entry for it.
    if name == "component" and value not in landscape.selected_components:
        raise BadInput(f"component must be one of {landscape.selected_components}")
    if name == "phase" and value not in PHASES:
        raise BadInput(f"phase must be one of {list(PHASES)}")
    return value


# One keeper object per pair of floors, so two requests asking for the same
# thing send the same OBJECT — the closure is the cache key of
# Landscape._nearest_cache, which is what lets the /export after a /render
# answer from cache. Inline this into a per-request lambda and that cache
# silently dies. The phase comes in per call, not into the key.
@functools.lru_cache(maxsize=None)
def word_keeper_for(min_tweets, min_similarity):
    return lambda words, phase: covered_words(words, min_tweets, min_similarity,
                                              phase_filter(phase))


def nearest_kwargs(args):
    """The centroid and nearest-word arguments of Landscape.nearest and
    .render. Each floor is a value behind a checkbox, off unless its checkbox
    says otherwise."""
    kwargs = {name: read(args, name) for name in NEAREST}
    # The one search parameter whose size is the app's problem: an unclamped
    # topn reaches the whole 40k-word vocabulary, and the display grouping
    # over it is quadratic. 200 is what the box offers.
    kwargs["topn"] = min(max(kwargs["topn"], 0), 200)
    kwargs["min_count"] = read(args, "min_count") if read(args, "min_count_filter") else 1
    kwargs["min_similarity"] = read(args, "min_similarity") if read(args, "similarity_filter") else None
    kwargs["word_keeper"] = None
    if read(args, "tweet_filter"):
        kwargs["word_keeper"] = word_keeper_for(read(args, "word_min_tweets"),
                                                read(args, "word_min_similarity"))
    return kwargs


def drawn_kwargs(args):
    """The display arguments of Landscape.render; each label family's
    frequency floor is a value behind a checkbox, None with the box off."""
    kwargs = {name: read(args, name) for name in DRAWN}
    # matplotlib allocates the figure buffer without a size guard of its own:
    # scale 30 asks for 2.4 GB, scale 60 for 9.8 GB. 2.5 is what the box offers.
    kwargs["scale"] = min(max(kwargs["scale"], 0.5), 2.5)
    for family in ("strong", "extreme"):
        kwargs[f"{family}_min_count"] = (
            read(args, f"{family}_min_count")
            if read(args, f"{family}_min_count_filter") else None)
    return kwargs


# What every control starts on, by parameter name — PARAMS stripped of the
# parsers. The markup reads it as `d`.
DEFAULTS = {name: default for name, (_, default, _) in PARAMS.items()}


# The two sets of prompts the run pages offer, in tab order. Frames first:
# the frame analysis is what the project is being used to produce, the
# narratives are what it grew out of and the comparison baseline.
PROMPT_LABELS = {"frames": "Frames", "narratives": "Narratives"}

# Which set a page opens on for a browser that has never chosen.
DEFAULT_PROMPT_SET = "frames"


def prompt_sets(**per_set):
    """One page's prompt sets: the shared label of each, with whatever the
    page adds for it (`base`, where the set's runs live, for a page that has
    runs). Keyed and ordered by PROMPT_LABELS, so every page offers the choice
    in the same order under the same names."""
    return {key: {"label": label} | per_set.get(key, {})
            for key, label in PROMPT_LABELS.items()}


def page_context(params, **extra):
    """What a page renders from: its parameters, the agent boxes, and — as
    `page_json`, plus whatever `extra` the page adds — the one `PAGE` blob its
    static scripts read every server value through."""
    return dict(params=params, d=DEFAULTS, components=landscape.selected_components,
                phases=PHASES, run_phases=RUN_PHASES,
                default_summary_model=DEFAULT_SUMMARY_MODEL,
                default_acueval_model=DEFAULT_ACUEVAL_MODEL,
                default_grader_model=DEFAULT_GRADER_MODEL,
                page_json=dict(params=params,
                               default_agent={"model": DEFAULT_SUMMARY_MODEL},
                               default_acue={"model": DEFAULT_ACUEVAL_MODEL},
                               default_grader={"model": DEFAULT_GRADER_MODEL},
                               **extra))


def agent_config(payload, name):
    """The model the page asks for for one agent, as keyword arguments to its
    class; an empty box falls back to the role's default."""
    asked = (payload.get("agents") or {}).get(name) or {}
    fallback = (DEFAULT_SUMMARY_MODEL if "summary" in name
                else DEFAULT_ACUEVAL_MODEL if "acueval" in name
                else DEFAULT_GRADER_MODEL)
    return {"model": (asked.get("model") or "").strip() or fallback}


# Which of the page's model boxes each of a level's four roles is built from.
# The budget has no box: its only call is the rewrite, and shortening an
# answer is writing rather than judging, so it goes to the writer's model.
ROLE_BOX = {"summary": "summary", "acueval": "acueval",
            "rubric": "rubric", "budget": "summary"}

# The three grader roles of a level; the writer stands apart.
GRADERS = ("acueval", "rubric", "budget")


def role_config(payload, role):
    """The model one role of a level runs on — the page's box for that role,
    or the writer's where it has none."""
    return agent_config(payload, ROLE_BOX[role])


def num(value, default, cast=float):
    """One number out of a stored run setting, or its fallback when the page
    stored something unreadable — `read`'s tolerance, for the settings that
    travel in a task's JSON rather than in a query string."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def stored_retrieval(params):
    """What a result page's "tweets" buttons re-retrieve with: the run's own
    settings, speaker restriction included, so they show the tweets the
    summaries were actually written from."""
    return {"n_tweets": num(params.get("n_tweets"), 20, int),
            "min_similarity": num(params.get("min_similarity"), 0.3),
            "speakers": params["speakers"],
            "component": params["component"],
            "n_extreme": params["n_extreme"]}


def payload_strs(payload, *names):
    """The named fields of a JSON payload, one stripped string each."""
    return [(payload.get(name) or "").strip() for name in names]


def csv_response(text, filename):
    """One CSV download, named as the browser saves it."""
    return Response(text, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


def no_such(what):
    """The one 404 every missing run answers with."""
    return jsonify({"error": f"No such {what}."}), 404


# A grade shows green when it clears its scale's floor here, red when not.
# The `warranted` floor colours a refusal by whether declining was right — a
# different question from the two above it, never mixed with them.
VIEW_FLOORS = {"acue": 0.8, "rubric": 4, "warranted": 4}

# How many decimals a grade keeps in the CSVs and on the pages.
DECIMALS = 2


@app.template_global()
def col_class(column):
    """The class naming the column a cell stands in, so a page can take a
    column out of its table whole. The two heads spell a column differently
    ("RUBRIC comment" grouped, "RUBRIC_comment" flat) and answer the same
    class."""
    return "col-" + str(column).strip().lower().replace(" ", "_")


def display_name(column):
    """How a column or label is shown — the grade families in their proper
    spelling. The stored keys keep their own words."""
    return (column.replace("acue", "ACUEval").replace("rubric", "RUBRIC")
            .replace("warranted", "WARRANTED").replace("budget_words", "words"))


def grade_cls(family, value):
    """"ok"/"bad" for one grade against its family's floor, "" without a grade."""
    if value is None or value == "":
        return ""
    return "ok" if value >= VIEW_FLOORS[family] else "bad"
