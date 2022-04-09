from queue import Queue
from typing import Optional

from simpy import Resource


class Client:
    """CLIENT"""

    def __init__(
            self,
            provider: str,
            threads: Optional[Resource] = None,
            payload_queue: Optional[Queue] = None
    ):
        self.threads = threads
        self.provider = provider
        self.payload_queue = payload_queue
