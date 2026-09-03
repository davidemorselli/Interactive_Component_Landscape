# Web version of the Component Landscape notebook.
#
# Importing the route modules below runs the boot pipeline (webapp.state) and
# registers every route on the shared app (webapp.app):
#   landscape_page   the landscape page: /render, /words, /export*, /suggested, /token
#   narrative        /tweets, /narrative, /grade, /refine
#   bulk_page        the bulk-narratives page and its background tasks
#   task_views       a task's results as browser table and CSV
#   frames_page      the bulk-frames page: the bulk_page machinery, frame prompts
#   comparison_page  the LLM-comparison page and its runs
#   frame_comparison_page
#                    the frames LLM comparison: the comparison_page
#                    machinery, frame agents
#   prompts_page     /prompts — every prompt the app sends, to be read
#
#   cd src && uv run waitress-serve --port=8050 webapp.webapp:app

from webapp.app import app  # noqa: F401
from webapp import (bulk_page, comparison_page, frame_comparison_page,  # noqa: F401
                    frames_page, landscape_page, narrative, prompts_page,
                    task_views)

# The deploy workflow (.github/workflows/deploy.yml) greps the compose logs for
# this exact line to declare a deploy healthy — printed only now, with the boot
# done and every route registered. Keep both sides in sync.
print("Ready.", flush=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, threaded=True, debug=False)
