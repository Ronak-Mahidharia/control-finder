# BM25 checked against a score worked out by hand.
import math

from controlfinder.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords_and_numbers():
    assert tokenize("The 2 passwords, for multi-factor use") == ["passwords", "multi", "factor", "use"]


def test_hand_worked_score():
    docs = ["password password policy", "audit log", "password audit"]
    index = BM25(docs, k1=1.5, b=0.75)
    # "password" is in 2 of 3 documents: idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    idf = math.log(1.6)
    average = (3 + 2 + 2) / 3
    norm = 1 - 0.75 + 0.75 * 3 / average
    expected = idf * 2 * 2.5 / (2 + 1.5 * norm)  # document 0 has "password" twice
    assert math.isclose(index.score(["password"], 0), expected)


def test_ranking_and_ties():
    index = BM25(["password password policy", "audit log", "password audit"])
    assert [i for i, _ in index.search("password")] == [0, 2]  # document 1 scores 0 and is left out
    tied = BM25(["alpha beta", "alpha beta"])
    assert [i for i, _ in tied.search("alpha")] == [0, 1]  # ties keep catalog order
