import json

from typing import List

from loopixdes.model.mail import Mail


def load_dataset(file_name: str) -> List[Mail]:
    with open(file_name, 'r') as file:
        dataset = [Mail(**mail) for mail in json.load(file)]

    return dataset
