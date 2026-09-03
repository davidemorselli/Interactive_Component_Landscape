# Every prompt the app sends, one module per level.
#
# `narrative` and `narrative_merge` are the two levels of the narratives set,
# `frames` and `frames_merge` the two of the frames set; `frame_topics` holds
# the one pass over a merged analysis that follows them, and `refusals` the
# one line code reads back off an answer without knowing which prompt asked
# for it. Every other such line is written out where it is used.
#
# Nothing here calls a model or knows what one answers:
# llm.pipeline.levels names these seams, and the agents read them off
# the level they were built for.
