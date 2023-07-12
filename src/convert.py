from typing import Generator, List, Iterable, NamedTuple
from src.prm800k_record import PRMRecord
from collections import deque

class Sample(NamedTuple):
    instruction: str
    responses: List[str]
    next_response: str

class MalformedRecord(Exception): ...


def make_telescoping_conversation(conversation: PRMRecord) -> Generator[Sample, None, None]:
    instruction: str = conversation['question']['problem']
    steps: List[str] = []
    for step in conversation['label']['steps']:
        if step['chosen_completion'] is None:
            raise MalformedRecord('no human_completion was provided')
        steps.append(step['human_completion']['text'] if step['chosen_completion'] is None else step['completions'][step['chosen_completion']]['text'])
        *precursors, latest = steps
        yield Sample(
            instruction=instruction,
            responses=precursors,
            next_response=latest,
        )


def get_final_sample(samples: Iterable[Sample]) -> Sample:
    last_element, = deque(samples, 1)
    return last_element
