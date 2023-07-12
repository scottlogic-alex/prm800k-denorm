from typing import TypedDict, Optional, Any, Literal, List

class Question(TypedDict):
    problem: str
    ground_truth_answer: str

class HumanCompletion(TypedDict):
    text: str
    rating: None
    source: Literal['human']
    flagged: bool

class Completion(TypedDict):
    text: str
    rating: int
    flagged: bool

class Step(TypedDict):
    completions: List[Completion]
    human_completion: Optional[HumanCompletion]
    chosen_completion: Optional[int]

class Label(TypedDict):
    finish_reason: Literal['solution', 'give_up']
    total_time: int
    steps: List[Step]

class PRMRecord(TypedDict):
    labeler: str
    timestamp: str
    generation: Optional[Any]
    is_quality_control_question: bool
    is_initial_screening_question: bool
    question: Question
    label: Label
