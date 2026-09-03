# The Flask app and its JSON error answers, shared by every route module.

import hashlib
import traceback
from pathlib import Path

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from llm.pipeline.agent import AgentError

app = Flask(__name__)


@app.template_global()
def asset(name):
    """
    A /static URL carrying the file's own content hash. The CDN tunnel and
    the browser cache /static by name, so a deploy that changed only a script
    kept serving the old one against the new markup (the strong-word controls
    once reached the site permanently greyed out that way); a new file is a
    new URL, an unchanged one stays cached.
    """
    path = Path(app.static_folder) / name
    try:
        stamp = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    except OSError:  # a name that is not there answers 404 either way
        stamp = "0"
    return f"/static/{name}?v={stamp}"


class BadInput(ValueError):
    """A parameter the app will not act on, with a message written for the
    user. Answered as a 400 with that message, where anything else gets a
    traceback."""


@app.errorhandler(BadInput)
def _bad_input(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(AgentError)
def _agent_unavailable(error):
    # The message is written for the user — no API key configured, a model
    # past its output ceiling — so it goes out as it is.
    return jsonify({"error": str(error)}), 503


@app.errorhandler(Exception)
def _bug(error):
    """Anything else an endpoint raises is a bug, and the page shows its
    traceback as the JSON error it reads."""
    if isinstance(error, HTTPException):  # real 404s and 405s answer as themselves
        return error
    return jsonify({"error": traceback.format_exc()}), 500
