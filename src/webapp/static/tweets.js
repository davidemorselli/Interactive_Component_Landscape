// Re-run a stored run's retrieval for one word — the query the task view's
// dialog and the side-by-side cards both send. The tweets are not stored with
// a run: retrieval is deterministic, so the run's own settings (`retrieval`,
// webapp params.stored_retrieval) bring back the tweets its summaries were
// written from — as long as the corpus has not changed since.
async function fetchStoredTweets(retrieval, word, phase) {
  const params = new URLSearchParams({
    query: word, phase,
    n_tweets: retrieval.n_tweets, min_similarity: retrieval.min_similarity,
    speakers: retrieval.speakers, component: retrieval.component,
    n_extreme: retrieval.n_extreme,
  });
  return (await fetch("/tweets?" + params)).json();
}
