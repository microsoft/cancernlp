"""
Utilities for aligning pre-extracted tokens (strings) with the original text.  Current
implementation only supports direct string matches and replacement of double-quotes with
left and right double-ticks ("``" and "''").
"""
import re
from dataclasses import dataclass
from typing import Dict, List


class AlignmentError(Exception):
    """Error arising from token alignment mismatch."""


# these are not all of the Penn Treebank token substitutions, but seem to be all that is
# used in the JNLPBA data.
PTB_REPLACEMENTS = {"``": '"', "''": '"'}

# simple matching function for token alignment, does some basic replacements used in the
# JNLPBA datast


def _ptb_match(token, text):
    token = PTB_REPLACEMENTS.get(token, token)
    if text.startswith(token):
        return token
    return None


@dataclass
class CharSpan:
    """Simple data class representing a character span."""

    begin: int
    end: int
    token: str
    matched_text: str


def align_tokens(text: str, tokens: List[str], match_func=_ptb_match) -> List[CharSpan]:
    """Given raw text and a list of tokens, find the corresponding begin/end position of
    each token in the raw text.

    match_func can be given to refine how tokens are matched; it takes a token and raw
    text, and should return True if the raw text starts with the given token, and should
    return the matched raw text."""
    cur_text_idx = 0
    token_alignment = []
    for token in tokens:
        while cur_text_idx < len(text):
            match = match_func(token, text[cur_text_idx:])
            if match is not None:
                end_idx = cur_text_idx + len(match)
                token_alignment.append(
                    CharSpan(
                        begin=cur_text_idx, end=end_idx, token=token, matched_text=match
                    )
                )
                cur_text_idx = end_idx
                break
            cur_text_idx += 1

    if len(token_alignment) < len(tokens):
        missing_idx = len(token_alignment)
        missing_token = tokens[missing_idx]
        next_text = text[token_alignment[-1].end : token_alignment[-1].end + 20]
        raise AlignmentError(
            f"Unable to align tokens, token {missing_idx} ({missing_token}) not found. "
            f' Next text was "{next_text}..."'
        )

    return token_alignment


PTB_REGEX_SUBSTITUTIONS = {"-PRON-": "(?:s?he|hi[ms]|hers?)", "``": '"', "''": '"'}


def regex_align_tokens(
    text: str, tokens: List[str], substitution_list: Dict[str, str] = None
) -> List[CharSpan]:
    """
    Given raw text and a list of tokens, find the corresponding begin/end position of
    each token in the raw text.

    substitution_list gives a mapping from tokens to regex expressions they should
    match.  Note that this function builds large regular expressions, it should probably
    not be run on very long text.
    """
    if substitution_list is None:
        substitution_list = PTB_REGEX_SUBSTITUTIONS
    text = text.lower()
    formatted_tokens = []
    for token in tokens:
        if token in substitution_list:
            formatted_tokens.append(substitution_list[token])
        else:
            formatted_tokens.append(re.escape(token.lower()))
    pattern = r"\s*".join(["(" + _token + ")" for _token in formatted_tokens])

    pattern = r"^\s*" + pattern + r"\s*$"
    match = re.match(pattern=pattern, string=text)
    if not match:
        raise AlignmentError(f"Could not align tokens {tokens} with '{text}'")
    return align_tokens(text=text, tokens=match.groups())
