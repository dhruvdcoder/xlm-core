import re
from typing import Any, Dict

from transformers import PreTrainedTokenizerBase
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def detokenizer_lm1b_string(x: str) -> str:
    """Detokenizer for the LM1B dataset.
    Same as the one used in SEDD and MDLM.
    """
    # MAYBE: Fix the issues in the detokenizer. Run it on the `detok_test_set.txt` to see the issues.
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(
        r" \'(\w+)", r"'\1", x
    )  # remove extra space before single quotes like "they 've"
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def preprocess_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    example["text"] = detokenizer_lm1b_string(example["text"])
    example["token_ids"] = tokenizer.encode(  # type: ignore
        example["text"],
        add_special_tokens=False,
    )
    return example
