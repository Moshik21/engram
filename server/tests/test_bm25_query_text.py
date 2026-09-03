"""BM25 lanes drop function words: they cost posting-list walks and carry no signal."""

from engram.storage.helix.search import bm25_query_text


def test_function_words_are_dropped() -> None:
    assert (
        bm25_query_text("what is the flip condition for usage ranking")
        == "flip condition usage ranking"
    )
    assert bm25_query_text("why was Thompson sampling removed?") == "Thompson sampling removed?"


def test_all_stopwords_keeps_the_original() -> None:
    assert bm25_query_text("what is it") == "what is it"
    assert bm25_query_text("") == ""
