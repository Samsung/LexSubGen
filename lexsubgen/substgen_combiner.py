from typing import List, Optional, Tuple, Any
from itertools import chain

class SubstituteGeneratorsCombiner:
    def __init__(self, subst_generators: List):
        self.subst_generators = subst_generators

    def generate_substitutes(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos: Optional[List[str]] = None,
        return_probs: bool = False,
        target_lemmas: Optional[List[str]] = None
    ) -> Tuple[List[List[str]], Any]:
        """
        Generates substitutes for a given batch of instances.

        Args:
            sentences: list of contexts
            target_ids: list of target indexes
            target_pos: list of target word pos tags
            return_probs: return substitute probabilities if True
            target_lemmas: list of target lemmas

        Returns:
            substitutes, vocabulary and optionally substitute probabilities
        """
        # TODO: combine probs from different generators
        assert not return_probs

        substitutes = []
        for substgen in self.subst_generators:
            substs, _ = substgen.generate_substitutes(
                sentences,
                target_ids,
                target_pos=target_pos,
                return_probs=False,
                target_lemmas=target_lemmas,
            )
            substitutes.append(substs)

        combined_substitutes = []
        for substs in zip(*substitutes):
            combined_substitutes.append(list(chain(*substs)))

        return combined_substitutes, None
