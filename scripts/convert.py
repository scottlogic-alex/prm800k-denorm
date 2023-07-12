import json
from dataclasses import dataclass
from typing import TypeAlias, Generator, Any

Conversation: TypeAlias = Any


@dataclass
class Sample:
    instruction: str
    steps: str


def make_telescoping_conversation(conversation: Conversation) -> Generator[Sample, None, None]:
    instruction = conversation['question']['problem']
    steps = []
    for step in conversation['label']['steps']:
        steps.append(step['completions'][step['chosen_completion']]['text'] if step['chosen_completion'] else step['human_completion']['text'])
        yield Sample(
            instruction=instruction,
            steps=json.dumps(steps)
        )


def make_monologue(conversation: Conversation) -> Sample:
    return Sample(
        instruction=conversation['question']['problem'],
        steps=json.dumps([step['completions'][step['chosen_completion']]['text'] if step['chosen_completion'] else step['human_completion']['text']
                          for step in conversation['label']['steps']])
    )
