"""
TextCorrector Module — Lightweight OCR Post-Processor
======================================================

Pure retrieval-based correction. No neural reranking.
Fast, deterministic, and preserves correct words.

Dependency:
    pip install wordfreq rapidfuzz
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

from rapidfuzz.distance import Levenshtein
from wordfreq import iter_wordlist, zipf_frequency


class TextCorrector:
    """
    Corrects OCR word-level errors using a frequency-aware,
    length-indexed lexical retrieval system.
    """

    # --------------------------------------------------------------------- #
    # Constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        max_length_diff: int = 3,
        max_edit_distance: int = 5,
        min_zipf: float = 2.5,
        debug: bool = False,
        min_char_overlap: float = 0.3,
        min_bigram_sim: float = 0.2,
        common_word_threshold: float = 5.5,
    ):
        self.max_length_diff = max_length_diff
        self.max_edit_distance = max_edit_distance
        self.min_zipf = min_zipf
        self.debug = debug
        self.min_char_overlap = min_char_overlap
        self.min_bigram_sim = min_bigram_sim
        self.common_word_threshold = common_word_threshold

        # Load filtered dictionary with length-based indexing
        self.dictionary: Set[str]
        self.word_freq: Dict[str, float]
        self._length_index: Dict[int, List[str]]
        self.dictionary, self.word_freq, self._length_index = self._load_dictionary()

        # Tokenization: words, numbers, paragraph breaks, newlines, spaces, punctuation
        self._token_pattern = re.compile(
            r"[a-zA-Z]+|\d+|\n{2,}|\n|[^\S\n]+|[^a-zA-Z\d\s]"
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def correct(self, text: str, ocr_tokens: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Perform the full correction pipeline.
        """
        normalized = self._normalize_text(text)
        tokens = self._tokenize(normalized)
        corrected_tokens = self._correct_tokens(tokens, ocr_tokens)
        reconstructed = self._reconstruct(corrected_tokens)
        cleaned = self._cleanup_text(reconstructed)
        return cleaned

    # --------------------------------------------------------------------- #
    # Pipeline Stages
    # --------------------------------------------------------------------- #
    def _normalize_text(self, text: str) -> str:
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = text.replace("\r", "")
        text = re.sub(r"\s+([.,;:!?)}\]])", r"\1", text)
        text = re.sub(r"([({\[])\s+", r"\1", text)
        return text.strip()

    def _tokenize(self, text: str) -> List[str]:
        return self._token_pattern.findall(text)

    def _reconstruct(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def _cleanup_text(self, text: str) -> str:
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\s+([.,;:!?)}\]])", r"\1", text)
        text = re.sub(r"([({\[])\s+", r"\1", text)
        text = re.sub(r"[“”]", '"', text)
        text = re.sub(r"[‘’]", "'", text)
        return text.strip()

    # --------------------------------------------------------------------- #
    # Token Correction
    # --------------------------------------------------------------------- #
    def _correct_tokens(
        self,
        tokens: List[str],
        ocr_tokens: Optional[List[Dict[str, str]]] = None,
    ) -> List[str]:
        corrected = []
        word_idx = 0
        for idx, token in enumerate(tokens):
            meta = None
            if ocr_tokens is not None and word_idx < len(ocr_tokens):
                meta = ocr_tokens[word_idx]

            if self._is_correctable(token, meta):
                corrected.append(self._correct_token(token, idx, meta))
                word_idx += 1
            else:
                corrected.append(token)
                if token.isalpha() and len(token) > 2:
                    word_idx += 1
        return corrected

    def _is_correctable(
        self,
        token: str,
        ocr_metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        if not token.isalpha() or len(token) <= 2:
            return False
        return True

    def _dynamic_max_edit(self, word_len: int) -> int:
        """Length-aware edit distance ceiling."""
        if word_len <= 4:
            return min(2, self.max_edit_distance)
        elif word_len <= 7:
            return min(3, self.max_edit_distance)
        else:
            return min(5, self.max_edit_distance)

    def _correct_token(
        self,
        word: str,
        index: int,
        ocr_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        lower = word.lower()

        # -----------------------------------------------------------------
        # 1. Known dictionary word → always keep it
        # -----------------------------------------------------------------
        if lower in self.dictionary:
            freq = self.word_freq[lower]
            if self.debug:
                print(f"[KEEP] '{word}'  (zipf={freq:.2f}, in-dictionary)")
            return word

        # -----------------------------------------------------------------
        # 2. OOV or very rare word → retrieve best dictionary candidate
        # -----------------------------------------------------------------
        candidates = self._candidate_pool(lower)
        if not candidates:
            if self.debug:
                print(f"[SKIP] '{word}' — no candidates")
            return word

        best = candidates[0][0]

        if self.debug:
            print(f"\n{'='*60}")
            print(f"OCR WORD:       '{word}'")
            print(f"CANDIDATE POOL: {[(c, f'{s:.2f}') for c, s in candidates]}")
            print(f"FINAL WINNER:   '{best}'")
            print(f"{'='*60}")

        return self._apply_capitalization(word, best)

    # --------------------------------------------------------------------- #
    # Candidate Retrieval
    # --------------------------------------------------------------------- #
    def _candidate_pool(self, ocr_word: str) -> List[Tuple[str, float]]:
        """
        Search length-indexed buckets.
        Score = edit_dist * 10 - prefix_len * 2 - suffix_len - min(freq,7)*1.5
        Lower score = better candidate.
        """
        ocr_len = len(ocr_word)
        max_dist = self._dynamic_max_edit(ocr_len)
        min_len = max(1, ocr_len - self.max_length_diff)
        max_len = ocr_len + self.max_length_diff

        candidates: List[Tuple[str, float]] = []

        for length in range(min_len, max_len + 1):
            if length not in self._length_index:
                continue
            for candidate in self._length_index[length]:
                edit_dist = Levenshtein.distance(ocr_word, candidate)
                if edit_dist > max_dist:
                    continue

                overlap = self._char_overlap(ocr_word, candidate)
                if overlap < self.min_char_overlap:
                    continue

                bigram = self._bigram_similarity(ocr_word, candidate)
                if bigram < self.min_bigram_sim:
                    continue

                score = self._retrieval_score(ocr_word, candidate, edit_dist)
                candidates.append((candidate, score))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1])
        return candidates[:5]

    def _retrieval_score(
        self,
        ocr_word: str,
        candidate: str,
        edit_dist: int,
    ) -> float:
        """
        Lightweight scoring.  Lower is better.
        """
        max_len = max(len(ocr_word), len(candidate), 1)

        # Prefix match
        prefix_len = 0
        for a, b in zip(ocr_word, candidate):
            if a == b:
                prefix_len += 1
            else:
                break

        # Suffix match
        suffix_len = 0
        for a, b in zip(reversed(ocr_word), reversed(candidate)):
            if a == b:
                suffix_len += 1
            else:
                break

        freq = self.word_freq.get(candidate, 0.0)

        score = edit_dist * 10
        score -= prefix_len * 2
        score -= suffix_len
        score -= min(freq, 7.0) * 1.5
        return score

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _char_overlap(self, w1: str, w2: str) -> float:
        s1, s2 = set(w1), set(w2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def _bigram_similarity(self, w1: str, w2: str) -> float:
        def bigrams(s):
            return {s[i : i + 2] for i in range(len(s) - 1)}

        b1, b2 = bigrams(w1), bigrams(w2)
        if not b1 or not b2:
            return 0.0
        inter = len(b1 & b2)
        return (2 * inter) / (len(b1) + len(b2))

    def _apply_capitalization(self, original: str, corrected: str) -> str:
        if original.isupper():
            return corrected.upper()
        if original and original[0].isupper():
            return corrected[0].upper() + corrected[1:]
        return corrected

    # --------------------------------------------------------------------- #
    # Dictionary Loading
    # --------------------------------------------------------------------- #
    def _load_dictionary(self) -> Tuple[Set[str], Dict[str, float], Dict[int, List[str]]]:
        words: Set[str] = set()
        freqs: Dict[str, float] = {}
        index: Dict[int, List[str]] = defaultdict(list)

        for w in iter_wordlist("en"):
            if not w.isalpha():
                continue
            if len(w) < 3:
                continue

            freq = zipf_frequency(w, "en")
            if freq < self.min_zipf:
                continue

            words.add(w)
            freqs[w] = freq
            index[len(w)].append(w)

        if self.debug:
            print(f"Loaded {len(words)} words from wordfreq.")

        return words, freqs, dict(index)


# ------------------------------------------------------------------------- #
# Example Usage
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    corrector = TextCorrector(debug=True)

    # Test 1: Clean OCR (should NOT be modified)
    clean = """This is a lot of 12 point text to test the
ocr code and see if it works on all types
of file format.
The quick brown dog jumped over the
lazy fox. The quick brown dog jumped
over the lazy fox. The quick brown dog
jumped over the lazy fox. The quick
brown dog jumped over the lazy fox."""

    result_clean = corrector.correct(clean)
    print(f"\n{'='*70}")
    print(f"CLEAN OCR:\n{clean}")
    print(f"\nCORRECTED:\n{result_clean}")
    print(f"{'='*70}")

    # Test 2: Corrupted OCR (should be fixed)
    raw_text = "What makea people smaut, cusions, alect, absesrant, competent, confident, resoncefrl, pesistent-in the broadest and beat sense intelligent - is rot havis access to more and more learning place, resonce, and Specialists, but beig able in their lives to do a wide varicty of interesting thiig Hat matte, things that challege their ingenuity, skill gudgenect, and that make an obvions diffeunce in Heir lite and the lives f people around them John Holt from Teach youm Com Sasden Dold 1928/01 abromdrorl"
    result_corrupt = corrector.correct(raw_text)
    print(f"\n{'='*70}")
    print(f"CORRUPTED OCR:\n{raw_text}")
    print(f"\nCORRECTED:\n{result_corrupt}")
    print(f"{'='*70}")