# ============================================================================
# Keyword search with Okapi BM25, the standard ranking behind most search engines.
#
# A control scores higher when it contains the question's words, especially
# rare words, and when those words make up more of its (shorter) text.
#   k1 = how quickly repeating a word stops adding to the score (1.5)
#   b  = how much a long control is penalized for its length (0.75)
# Plain Python, so every step of the score can be read and tested.
# ============================================================================
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
# Words too common to tell controls apart.
STOP = frozenset("""a an and are as at be by for from has have in into is it its of on or
that the their this to with within""".split())


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and not t.isdigit()]


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / len(self.lengths)
        n = len(self.counts)
        document_frequency = Counter(term for c in self.counts for term in c)
        # The "+1" keeps every weight positive, even for a word that appears in most controls.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in document_frequency.items()}

    def score(self, query_terms, i):
        counts, length = self.counts[i], self.lengths[i]
        total = 0.0
        for term in query_terms:
            f = counts.get(term)
            if f:
                norm = 1 - self.b + self.b * length / self.average_length
                total += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
        return total

    def search(self, query, k=10):
        """The top k documents as (index, score), best first. Ties keep catalog order."""
        terms = tokenize(query)
        scores = [(i, self.score(terms, i)) for i in range(len(self.counts))]
        scores = [s for s in scores if s[1] > 0]
        scores.sort(key=lambda s: (-s[1], s[0]))
        return scores[:k]
