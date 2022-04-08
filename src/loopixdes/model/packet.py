from numpy import array
from typing import List
from typing import Tuple


class Packet:
    """PACKET"""

    def __init__(
            self,
            path: List[Tuple[str, float]],
            split: int,
            msg_id: str,
            sender: str,
            of_type: str,
            num_splits: int,
            expected_delay: float
    ):
        self.path = path
        self.split = split
        self.msg_id = msg_id
        self.sender = sender
        self.of_type = of_type
        self.num_splits = num_splits
        self.expected_delay = expected_delay

        if of_type != 'LOOP_MIX':
            self.dist = array([0.0, 0.0, 1.0])
