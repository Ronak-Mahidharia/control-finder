# ============================================================================
# Following NIST's links between controls. Each SP 800-53 control lists the
# controls it's related to (3,512 "related" links in Rev 5.2.0's catalog), and
# a question that needs one control often needs the ones it points to. So the
# hybrid search's top results lend part of their score to the controls they
# link to, and the list is sorted again.
#
# - Only links between active controls and enhancements count.
# - "one way" follows a link from the control that lists it; "both ways" also
#   from the control it names.
# - A seed is one of the top `seeds` results; each linked control gains
#   `weight` times the seed's fusion score, once for every seed that links to it.
# ============================================================================
from controlfinder.catalog import SP_800_53, read_json, walk_controls


def related_links(active_ids, both_ways=False):
    """Each active control's linked active controls, from the catalog's "related" links."""
    links = {cid: set() for cid in active_ids}
    for group in read_json(SP_800_53)["catalog"]["groups"]:
        for control in walk_controls(group):  # every control and enhancement, at any depth
            if control["id"] not in links:
                continue
            for link in control.get("links", []) or []:
                target = link.get("href", "").lstrip("#")
                if link.get("rel") == "related" and target in links and target != control["id"]:
                    links[control["id"]].add(target)
                    if both_ways:
                        links[target].add(control["id"])
    return links


def fusion_scores(rankings, k=60, depth=100, weights=None):
    """Reciprocal rank fusion's score for every id, as controlfinder.fusion.rrf ranks them."""
    weights = weights or [1.0] * len(rankings)
    scores, best_rank = {}, {}
    for ranking, weight in zip(rankings, weights):
        for rank, control_id in enumerate(ranking[:depth], start=1):
            scores[control_id] = scores.get(control_id, 0.0) + weight / (k + rank)
            best_rank[control_id] = min(best_rank.get(control_id, rank), rank)
    return scores, best_rank


def expand(scores, best_rank, links, seeds, weight):
    """The fused list with each top-`seeds` result lending `weight` of its score to its linked controls."""
    order = sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))
    boosted = dict(scores)
    for seed in order[:seeds]:
        for linked in links.get(seed, ()):
            boosted[linked] = boosted.get(linked, 0.0) + weight * scores[seed]
    far = len(order) + 1  # a control only a link brought in ranks after every searched one on a tie
    return sorted(boosted, key=lambda c: (-boosted[c], best_rank.get(c, far), c))
