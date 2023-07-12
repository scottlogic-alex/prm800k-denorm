from typing import Generator, List, Iterable, NamedTuple, Optional
from src.prm800k_record import PRMRecord
from collections import deque

class Sample(NamedTuple):
    instruction: str
    responses: List[str]
    next_response: str
    answer: Optional[str]

class GiveUp(Exception): ...

answer_delimiter = '\n\n# Answer\n\n'

def make_telescoping_conversation(conversation: PRMRecord) -> Generator[Sample, None, None]:
    instruction: str = conversation['question']['problem']
    steps: List[str] = []
    for step in conversation['label']['steps']:
        is_last: bool = step is conversation['label']['steps'][-1]
        is_solution: bool = is_last and conversation['label']['finish_reason'] == 'solution'
        if step['chosen_completion'] is None and step['human_completion'] is None:
            assert conversation['label']['finish_reason'] == 'give_up'
            raise GiveUp
        text: str = step['human_completion']['text'] if step['chosen_completion'] is None else step['completions'][step['chosen_completion']]['text']
        if is_solution:
            assert answer_delimiter in text
            next_response, answer = text.split(answer_delimiter, maxsplit=1)
        else:
            next_response, answer = text, None
        steps.append(next_response)
        yield Sample(
            instruction=instruction,
            responses=steps[:-1],
            next_response=next_response,
            answer=answer,
        )


def get_final_sample(samples: Iterable[Sample]) -> Sample:
    last_element, = deque(samples, 1)
    return last_element
