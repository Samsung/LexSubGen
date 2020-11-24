from enum import Enum, auto
from functools import lru_cache
from typing import Optional

from nltk.corpus import wordnet as wn


class Relation(Enum):
    """
    Class that contains all the considered WordNet relation types.
    """

    synonym = auto()
    co_hyponym = auto()
    co_hyponym_3 = auto()
    transitive_hypernym = auto()
    transitive_hyponym = auto()
    direct_hypernym = auto()
    direct_hyponym = auto()
    similar_to = auto()
    no_path = auto()
    unknown_relation = auto()
    unknown_word = auto()
    mwe = auto()
    same = auto()
    target_form = auto()
    meronym = auto()
    holonym = auto()
    entailment = auto()
    anti_entailment = auto()


to_wordnet_pos = {
    "n": wn.NOUN,
    "a": wn.ADJ,
    "v": wn.VERB,
    "r": wn.ADV,
    "n.v": wn.VERB,
    "n.a": wn.ADJ,
    "J": wn.ADJ,
    "V": wn.VERB,
    "R": wn.ADV,
    "N": wn.NOUN,
}


def get_synsets(word: str, pos: Optional[str] = None):
    """
    Acquires synsets for a given word and optionally pos tag.

    Args:
        word: word
        pos: pos tag of a word (optional)

    Returns:
        list of WordNet synsets.
    """
    return wn.synsets(word, pos=pos)


@lru_cache(maxsize=8192)
def get_similar_tos(word: str, pos: Optional[str] = None):
    """
    Find `similar to` synsets for a given word and optionally synset.
    Works with adjectives.

    Args:
        word: word to be analyzed
        pos: pos tag of a word

    Returns:
        set of `simialr to` words
    """
    similar_to_synsets = [
        first_lvl_sn
        for tgt_sns in get_synsets(word, pos=pos)
        for first_lvl_sn in tgt_sns.similar_tos()
    ]
    similar_tos = {
        lemma
        for first_lvl_sn in similar_to_synsets
        for lemma in first_lvl_sn.lemma_names()
    }

    similar_tos = similar_tos.union(
        {
            lemma
            for first_lvl_sn in similar_to_synsets
            for second_lvl_sn in first_lvl_sn.similar_tos()
            for lemma in second_lvl_sn.lemma_names()
        }
    )

    return similar_tos


def get_holonyms(synset):
    """
    Acquires holonyms from a given synset.

    Args:
        synset: WordNet synset.

    Returns:
        set of holonyms
    """
    return set(
        synset.member_holonyms() + synset.substance_holonyms() + synset.part_holonyms()
    )


def get_meronyms(synset):
    """
    Acquires meronyms for a given synset.

    Args:
        synset: WordNet synset

    Returns:
        set of meronyms
    """
    return set(
        synset.member_meronyms() + synset.substance_meronyms() + synset.part_meronyms()
    )


def find_nearest_synsets(target_synsets, subst_synsets, pos: Optional[str] = None):
    """
    Finds nearest path between two lists of synsets (target word synsets and substitute word synsets),
    e.g. finds two synsets, one from the
    first list and one from another, distance between which are the shortest.

    Args:
        target_synsets: list of synsets of a target word
        subst_synsets: list of synsets of a substitute word
        pos: pos tag of a target word (optional)

    Returns:
        two closest synsets - one for target word and another for substitute.
    """
    # TODO: Parallelize processing
    dists = [
        (tgt_syn, sbt_syn, dist)
        for tgt_syn in target_synsets
        for sbt_syn in subst_synsets
        for dist in [tgt_syn.shortest_path_distance(sbt_syn)]
        if dist is not None
    ]

    if len(dists) == 0:
        return None, None

    tgt_sense, sbt_sense, _ = min(dists, key=lambda x: x[2])

    return tgt_sense, sbt_sense


@lru_cache(maxsize=262144)  # 2**18
def get_wordnet_relation(target: str, subst: str, pos: Optional[str] = None) -> str:
    """
    Finds WordNet relation between a target word and a substitute by analyzing
    their synsets. Optionally one could specify pos tag of the target word for
    more robust analysis.

    Args:
        target: target word
        subst: substitute
        pos: pos tag of the target word

    Returns:
        WordNet relation between the target word and a substitute.
    """
    if pos:
        pos = pos.lower()

    if pos is None:
        pos = wn.NOUN

    if len(subst.split(" ")) > 1:
        return Relation.mwe.name

    if target == subst:
        return Relation.same.name

    if set(wn._morphy(target, pos)).intersection(set(wn._morphy(subst, pos))):
        return Relation.target_form.name

    target_synsets = get_synsets(target, pos=pos)
    subst_synsets = get_synsets(subst, pos=pos)
    if len(subst_synsets) == 0:
        return Relation.unknown_word.name

    target_lemmas = {lemma for ss in target_synsets for lemma in ss.lemma_names()}
    subst_lemmas = {lemma for ss in subst_synsets for lemma in ss.lemma_names()}
    if len(target_lemmas.intersection(subst_lemmas)) > 0:
        return Relation.synonym.name

    if subst in get_similar_tos(target, pos):
        return Relation.similar_to.name

    tgt_sense, sbt_sense = find_nearest_synsets(target_synsets, subst_synsets, pos)

    if tgt_sense is None or sbt_sense is None:
        return Relation.no_path.name

    extract_name = lambda synset: synset.name().split(".")[0]
    tgt_name, sbt_name = extract_name(tgt_sense), extract_name(sbt_sense)

    target_holonyms = get_holonyms(tgt_sense)
    target_meronyms = get_meronyms(tgt_sense)

    if sbt_name in {lemma for ss in target_holonyms for lemma in ss.lemma_names()}:
        return Relation.holonym.name
    if sbt_name in {lemma for ss in target_meronyms for lemma in ss.lemma_names()}:
        return Relation.meronym.name

    target_entailments = {
        lemma for ss in tgt_sense.entailments() for lemma in ss.lemma_names()
    }
    if sbt_name in target_entailments:
        return Relation.entailment.name

    subst_entailments = {
        lemma for ss in sbt_sense.entailments() for lemma in ss.lemma_names()
    }
    if tgt_name in subst_entailments:
        return Relation.anti_entailment.name

    for common_hypernym in tgt_sense.lowest_common_hypernyms(sbt_sense):
        tgt_hyp_path = tgt_sense.shortest_path_distance(common_hypernym)
        sbt_hyp_path = sbt_sense.shortest_path_distance(common_hypernym)

        if tgt_hyp_path == 1 and sbt_hyp_path == 0:
            return Relation.direct_hypernym.name  # substitute is a hypernym of target
        elif tgt_hyp_path == 0 and sbt_hyp_path == 1:
            return Relation.direct_hyponym.name
        elif tgt_hyp_path > 1 and sbt_hyp_path == 0:
            return Relation.transitive_hypernym.name
        elif tgt_hyp_path == 0 and sbt_hyp_path > 1:
            return Relation.transitive_hyponym.name
        elif tgt_hyp_path == 1 and sbt_hyp_path == 1:
            return Relation.co_hyponym.name
        elif max(tgt_hyp_path, sbt_hyp_path) <= 3:
            return Relation.co_hyponym_3.name

    return Relation.unknown_relation.name
