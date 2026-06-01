from typing import Dict, List, Tuple


class ValidPairGenerator:
    """Generate type-dependent valid (i, j) pairs in peptide index space."""

    def __init__(
        self,
        type_list: List[str],
        rule: str = "default",
        min_sep: int = 1,
    ) -> None:
        self.type_list = type_list
        self.rule = rule
        self.min_sep = max(1, int(min_sep))
        self._cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def _all_pairs(self, n: int) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(n) for j in range(i + self.min_sep, n)]

    def _nterm_pairs(self, n: int) -> List[Tuple[int, int]]:
        if n < 2:
            return []
        return [(0, j) for j in range(1, n)]

    def _end_to_end(self, n: int) -> List[Tuple[int, int]]:
        if n < 2:
            return []
        return [(0, n - 1)]

    def _resolve_rule(self, type_name: str, n: int) -> List[Tuple[int, int]]:
        if self.rule == "default":
            if type_name == "HT":
                return self._end_to_end(n)
            if type_name == "SS":
                return self._all_pairs(n)
            if type_name == "ISO":
                return self._all_pairs(n)
            return self._all_pairs(n)
        if self.rule == "all_pairs":
            return self._all_pairs(n)
        if self.rule == "end_to_end":
            return self._end_to_end(n)
        if self.rule == "nterm":
            return self._nterm_pairs(n)
        if self.rule == "custom":
            raise ValueError("custom rule requires user-defined pairs.")
        raise ValueError(f"Unknown pair rule: {self.rule}")

    def get_pairs(self, c_idx: int, n: int) -> List[Tuple[int, int]]:
        key = (c_idx, n)
        if key in self._cache:
            return self._cache[key]
        if c_idx < 0 or c_idx >= len(self.type_list):
            raise IndexError(f"Type index {c_idx} out of range.")
        type_name = self.type_list[c_idx]
        pairs = self._resolve_rule(type_name, n)
        self._cache[key] = pairs
        return pairs

    @staticmethod
    def build_pair_index(pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        return {pair: idx for idx, pair in enumerate(pairs)}

    @staticmethod
    def topk_from_logits(
        pairs: List[Tuple[int, int]],
        logits,
        k: int,
    ) -> List[Tuple[int, int, float]]:
        if logits.numel() == 0:
            return []
        k = min(int(k), logits.numel())
        topk = logits.topk(k)
        results = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            i, j = pairs[idx]
            results.append((i, j, float(score)))
        return results
