from os import makedirs, listdir
from os.path import dirname, realpath, exists
from pathlib import Path
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import json
from typing import Iterable, NamedTuple, List, Optional
from logging import getLogger, Logger
from src.prm800k_record import PRMRecord
from src.convert import make_telescoping_conversation, make_critiques, Sample, GiveUp
import fnmatch

logger: Logger = getLogger(__file__)

stepwise_schema = pa.schema([
    ('instruction', pa.string()),
    ('responses', pa.list_(pa.string())),
    ('next_response', pa.string()),
    pa.field('answer', pa.string(), nullable=True),
    ('is_human_response', pa.bool_()),
])

answer_only_schema = pa.schema([
    ('instruction', pa.string()),
    ('responses', pa.list_(pa.string())),
    ('next_response', pa.string()),
    pa.field('answer', pa.string(), nullable=True),
])

critique_schema = pa.schema([
    ('instruction', pa.string()),
    ('responses', pa.list_(pa.string())),
    ('next_response', pa.string()),
    pa.field('answer', pa.string(), nullable=True),
    ('is_human_response', pa.bool_()),
    ('is_solution', pa.bool_()),
    ('is_preferred_response', pa.bool_()),
    ('rating', pa.int8()),
])

class StepwiseBatch(NamedTuple):
    instructions: List[str]
    response_lists: List[List[str]]
    next_responses: List[str]
    answers: List[bool]
    is_human_response: List[bool]

class AnswerOnlyBatch(NamedTuple):
    instructions: List[str]
    response_lists: List[List[str]]
    next_responses: List[str]
    answers: List[bool]

class CritiqueBatch(NamedTuple):
    instructions: List[str]
    response_lists: List[List[str]]
    next_responses: List[str]
    answers: List[bool]
    is_human_response: List[bool]
    is_solution: List[bool]
    is_preferred_response: List[bool]
    ratings: List[int]

if __name__ == '__main__':
    script_dir = Path(dirname(realpath(__file__)))
    repo_root: Path = script_dir.parent
    out_dir = repo_root.joinpath('out')
    makedirs(out_dir, exist_ok=True)

    data_dir: Path = repo_root.joinpath('prm800k/data')
    assert exists(data_dir), 'Expected dir prm800k/data to exist -- you are expected to copy this in yourself from https://github.com/Openai/Prm800k. See prm800k/README.md for details.'

    for data_file in fnmatch.filter(listdir(data_dir), '*.jsonl'):
        data_stem = Path(data_file).stem
        in_path_jsonl: Path = data_dir.joinpath(data_file)

        out_all: Path = out_dir.joinpath(f'{data_stem}.parquet')
        out_all.unlink(missing_ok=True)
        out_answer_only: Path = out_dir.joinpath(f'{data_stem}.answer_only.parquet')
        out_answer_only.unlink(missing_ok=True)
        out_critique: Path = out_dir.joinpath(f'{data_stem}.critique.parquet')
        out_critique.unlink(missing_ok=True)

        with (open(in_path_jsonl, 'r') as file,
            ParquetWriter(str(out_all), schema=stepwise_schema) as stepwise_writer,
            ParquetWriter(str(out_answer_only), schema=answer_only_schema) as answer_only_writer,
            ParquetWriter(str(out_critique), schema=critique_schema) as out_critique_writer):
            # limit to reading 10 lines of the JSONL for now
            for line_ix, (line, _) in enumerate(zip(file.readlines(), range(3))):
                js: PRMRecord = json.loads(line)

                samples: Iterable[Sample] = make_telescoping_conversation(js)
                batch = StepwiseBatch([], [], [], [], [])
                instructions, response_lists, next_responses, answers, is_human_responses = batch
                final_sample: Optional[Sample] = None
                try:
                    for sample_ix, sample in enumerate(samples):
                        instruction, responses, next_response, answer, is_human_response = sample
                        instructions.append(instruction)
                        response_lists.append(responses)
                        next_responses.append(next_response)
                        answers.append(answer)
                        is_human_responses.append(is_human_response)
                        if answer is not None:
                            final_sample = sample
                except GiveUp as e:
                    logger.warning(f'Record at line {line_ix} gave up, at conversation step {sample_ix+1}')
                
                if batch.instructions:
                    table = pa.Table.from_arrays(list(batch), schema=stepwise_schema)
                    stepwise_writer.write_table(table)

                if final_sample is not None:
                    instruction, responses, next_response, answer, _ = final_sample
                    batch = AnswerOnlyBatch([instruction], [responses], [next_response], [answer])
                    table = pa.Table.from_arrays(list(batch), schema=answer_only_schema)
                    answer_only_writer.write_table(table)
                
                samples: Iterable[Sample] = make_critiques(js)
                batch = CritiqueBatch([], [], [], [], [], [], [], [])
                instructions, response_lists, next_responses, answers, is_human_responses, is_solutions, is_preferred_responses, ratings = batch
                for sample_ix, sample in enumerate(samples):
                    instruction, responses, next_response, answer, is_human_response, is_solution, is_preferred_response, rating = sample
                    instructions.append(instruction)
                    response_lists.append(responses)
                    next_responses.append(next_response)
                    answers.append(answer)
                    is_human_responses.append(is_human_response)
                    is_solutions.append(is_solution)
                    is_preferred_responses.append(is_preferred_response)
                    ratings.append(rating)
                table = pa.Table.from_arrays(list(batch), schema=critique_schema)
                out_critique_writer.write_table(table)