from typing import Self

from eth_account._utils.signing import to_standard_v
from eth_keys.datatypes import Signature as EthSignature
from py_flare_common.fsp.messaging.types import Signature as ParsedSignature


class Signature(EthSignature):
    @classmethod
    def from_parsed_signature(cls, s: ParsedSignature) -> Self:
        return cls(
            vrs=(
                to_standard_v(int(s.v, 16)),
                int(s.r, 16),
                int(s.s, 16),
            )
        )
