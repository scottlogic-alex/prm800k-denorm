from typing import Generator, List, Iterable, NamedTuple, Optional
from src.prm800k_record import PRMRecord, Completion, FinishReason
from collections import deque

class Sample(NamedTuple):
    instruction: str
    responses: List[str]
    next_response: str
    answer: Optional[str]
    is_human_response: bool

class CritiqueSample(NamedTuple):
    instruction: str
    responses: List[str]
    next_response: str
    answer: Optional[str]
    is_human_response: bool
    is_solution: bool
    is_preferred_response: bool
    # -1, 0, 1
    rating: int

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
        preferred_completion: Completion = step['human_completion'] if step['chosen_completion'] is None else step['completions'][step['chosen_completion']]
        is_human_response: bool = preferred_completion is step['human_completion']
        completion_text: str = preferred_completion['text']
        if is_solution:
            assert answer_delimiter in completion_text
            next_response, answer = completion_text.split(answer_delimiter, maxsplit=1)
        else:
            next_response, answer = completion_text, None
        steps.append(next_response)
        yield Sample(
            instruction=instruction,
            responses=steps[:-1],
            next_response=next_response,
            answer=answer,
            is_human_response=is_human_response,
        )

def make_critiques(conversation: PRMRecord) -> Generator[CritiqueSample, None, None]:
    instruction: str = conversation['question']['problem']
    steps: List[str] = []
    for step in conversation['label']['steps']:
        is_last: bool = step is conversation['label']['steps'][-1]
        has_solution: bool = is_last and conversation['label']['finish_reason'] == 'solution'
        preferred_completion: Completion = step['human_completion'] if step['chosen_completion'] is None else step['completions'][step['chosen_completion']]
        preferred_completion_text: str = preferred_completion['text']
        steps.append(preferred_completion_text)
        completions: List[Completion] = step['completions'] if step['human_completion'] is None else [
            *step['completions'],
            step['human_completion']
        ]
        for completion in completions:
            completion_text: str = completion['text']
            if answer_delimiter in completion_text:
                next_response, answer = completion_text.split(answer_delimiter, maxsplit=1)
            else:
                next_response, answer = completion_text, None
            is_human_response = completion is step['human_completion']
            yield CritiqueSample(
                instruction=instruction,
                responses=steps[:-1],
                next_response=next_response,
                answer=answer,
                is_human_response=is_human_response,
                is_solution=has_solution and completion is preferred_completion,
                is_preferred_response=completion is preferred_completion,
                rating=completion['rating'],
            )


def get_final_sample(samples: Iterable[Sample]) -> Sample:
    last_element, = deque(samples, 1)
    return last_element
