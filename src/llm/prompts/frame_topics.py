# The frame topic pass over a merged analysis: its frames are put in an order
# and given headings, and nothing else. Not part of the write/grade/refine
# pipeline — a blind second look, graded by nobody, with only the frame count
# read back off the answer (webapp.frames_page.topic_merge). Every frame block
# comes through to the character, which is what lets the grouped analysis keep
# the grades of the one it grouped: the claims are the same claims. A topic
# names the subject its frames are about, never a position on it — a heading
# that took one side would file the dispute under half of itself.


FRAME_TOPICS_SYSTEM_PROMPT = (
    "You are a discourse analyst arranging one merged frame analysis by "
    "subject, following Entman's (1993) definition of framing.")


def frame_topics_prompt(analysis: str) -> list:
    """The conversation that asks for one merged frame analysis in topic
    order: the same frames, to the character, under headings naming what each
    group is about — or the NO TOPICS line when the frames share no subject.

    Its answer is taken as it comes: it is asked for headings and an order and
    nothing else, and only the number of frames it hands back is checked."""

    user = f"""You are given ONE merged frame analysis. Put its frames in
topic order and head each group, and change nothing else.

WHAT A TOPIC IS
A topic is the subject a group of frames is about — the thing they are
all talking about, named in a short noun phrase. It is not a claim, not
a judgment, and not a problem definition: nobody could agree or disagree
with a topic, because it takes no position.
- "Vaccine mandates" is a topic. "Vaccine mandates are coercive" is a
  position, and belongs in a frame, not in a heading.
- "The vaccine rollout" is a topic. "The rollout is going badly" is a
  position.
Frames that CONTRADICT each other belong under the SAME topic. Two
frames that define the same subject in opposite ways are that subject
being disputed, and that dispute is exactly what a reader has come to
see: splitting them apart, or heading them with one side's view, hides
it. A topic holding two frames that disagree is the pass working, not a
mistake.

WHAT YOU MAY DO
1. REORDER the frames, so that frames about one subject stand together.
2. WRITE a TOPIC line above each group, naming that subject.
That is all you may do.

WHAT YOU MAY NOT DO
- Never change a frame. Every line of every frame block — the FRAME line
  with its problem definition and its attestation IDs, the function
  headings, every "- " value, every "none" — comes through EXACTLY as it
  stands, to the character. You are not rewriting, tidying, shortening,
  merging or splitting anything.
- Never drop a frame, and never write one. Every frame in the analysis
  appears in your answer exactly once, and nothing appears that was not
  in the analysis.
- Never leave a frame outside a topic. Your answer opens with a TOPIC
  line, and every frame stands under one.
- Never write a topic that holds no frame.
- Never write anything else. No preamble, no commentary between groups,
  no summary at the end, no code fences. Your answer is TOPIC lines and
  frame blocks and nothing besides.
- Never take a side in a heading, and never state a cause, a judgment or
  a remedy in one. If a heading could be contradicted, it is a claim and
  not a topic.

HOW MANY TOPICS
FEW. You are classifying the frames into a set of subjects, not labelling
them one by one. Aim for a handful of topics that between them cover
everything: think three or four for a dozen frames, and rarely more than
five however many there are. Every topic should hold more than one frame
wherever the frames allow it.

A heading over a single frame is a last resort, not a default. Before you
write one, look again at the topics you already have and ask whether that
frame belongs under one of them at a slightly wider reading of it — it
usually does. A frame about how a policy is enforced and a frame about
whether the policy should exist are both about that policy. A frame about
a product's safety and a frame about who is to blame for the failure are
both about that product. Widen the heading and put them together.

What you must not do is widen a heading until it says nothing. A topic
covering every frame in the analysis — "COVID-19", "the vaccine debate" —
has classified nothing, and is as useless as a heading per frame. The
right answer is in between: the smallest set of topics that still tells a
reader which frames are about the same thing.

When the frames genuinely share no subject and every group would hold
exactly one, answer with one line and nothing else:

NO TOPICS: <one sentence saying why these frames fall under no topics>

EXAMPLE
Input analysis (this example's content must never appear in your output
for other corpora):

FRAME: Load-shedding is destroying small businesses. [a1]
CAUSAL INTERPRETATION:
- Hours without power each day make trading impossible.
MORAL EVALUATION:
- The utility is ruining people who did nothing wrong.
TREATMENT RECOMMENDATION:
- Compensate the businesses that have been shut down.

FRAME: Official information about the blackouts cannot be trusted. [a1, a4]
CAUSAL INTERPRETATION:
- The institution times and relabels events to serve its own interests.
MORAL EVALUATION:
- The institution is dishonest with the public.
TREATMENT RECOMMENDATION: none

FRAME: Load-shedding is a necessary measure that keeps the grid from collapsing entirely. [a2]
CAUSAL INTERPRETATION:
- Shedding load protects the rest of the network from a total shutdown.
MORAL EVALUATION:
- The engineers holding the system together deserve credit.
TREATMENT RECOMMENDATION: none

Expected output:

TOPIC: The effects of load-shedding

FRAME: Load-shedding is destroying small businesses. [a1]
CAUSAL INTERPRETATION:
- Hours without power each day make trading impossible.
MORAL EVALUATION:
- The utility is ruining people who did nothing wrong.
TREATMENT RECOMMENDATION:
- Compensate the businesses that have been shut down.

FRAME: Load-shedding is a necessary measure that keeps the grid from collapsing entirely. [a2]
CAUSAL INTERPRETATION:
- Shedding load protects the rest of the network from a total shutdown.
MORAL EVALUATION:
- The engineers holding the system together deserve credit.
TREATMENT RECOMMENDATION: none

TOPIC: What the public is told about the blackouts

FRAME: Official information about the blackouts cannot be trusted. [a1, a4]
CAUSAL INTERPRETATION:
- The institution times and relabels events to serve its own interests.
MORAL EVALUATION:
- The institution is dishonest with the public.
TREATMENT RECOMMENDATION: none

---REASONING---
REASONING: The harm frame and the necessity frame are the same subject —
what load-shedding does — argued in opposite directions, so they stand
together under a heading that takes neither side. The trust frame is
about the account of the blackouts rather than the blackouts, and is a
subject of its own.

Why this output is correct:
- The two frames under the first heading contradict each other, and are
  grouped together for exactly that reason. "The effects of
  load-shedding" names what they argue about without saying who is
  right; "Load-shedding is harmful" would have been one of the two
  frames' positions wearing the hat of a heading.
- The frames moved, and nothing inside one did. Every FRAME line, every
  function heading, every value and every "none" is the analysis's own,
  character for character, and the attestation IDs are untouched.
- Three frames became two groups, not three. A heading per frame would
  have arranged nothing.
- Had the frames shared no subject, the whole answer would have been the
  NO TOPICS: line.

OUTPUT
Respond with the frames in topic order: a "TOPIC:" line naming a
subject, then the frame blocks belonging to it exactly as they came, then
the next "TOPIC:" line, and so on — then a line holding
"---REASONING---" and nothing else, then one line opening
"REASONING:" saying in two to four sentences what you grouped and why.
The reasoning is read on its own and is never part of the analysis:
nothing above the separator may refer to the topics, and nothing below
it may add to the analysis. No preamble, no commentary elsewhere, no code
fences. Or, when the frames fall into no groups, the
"NO TOPICS:" line alone and nothing else.

ANALYSIS:
{analysis}

ANSWER:
"""

    return [{"role": "system", "content": FRAME_TOPICS_SYSTEM_PROMPT},
            {"role": "user", "content": user}]

