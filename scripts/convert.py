from os import makedirs
from os.path import dirname, realpath, exists
from pathlib import Path
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import json
from typing import Iterable, NamedTuple, List
from logging import getLogger, Logger
from src.prm800k_record import PRMRecord
from src.convert import make_telescoping_conversation, Sample, GiveUp

logger: Logger = getLogger(__file__)

schema = pa.schema([
    ('instruction', pa.string()),
    ('responses', pa.list_(pa.string())),
    ('next_response', pa.string()),
])

class Batch(NamedTuple):
    instructions: List[str]
    response_lists: List[List[str]]
    next_responses: List[str]

if __name__ == '__main__':
    script_dir = Path(dirname(realpath(__file__)))
    repo_root: Path = script_dir.parent
    out_dir = repo_root.joinpath('out')
    makedirs(out_dir, exist_ok=True)

    data_stem = 'phase1_train'

    in_path_jsonl: Path = repo_root.joinpath(f'prm800k/data/{data_stem}.jsonl')
    assert exists(in_path_jsonl)

    out_path_parquet: Path = out_dir.joinpath(f'{data_stem}.parquet')
    out_path_parquet.unlink(missing_ok=True)

    with open(in_path_jsonl,'r') as file:
        with ParquetWriter(str(out_path_parquet), schema=schema) as writer:
            # limit to reading 2 lines of the JSONL for now
            for line_ix, (line, _) in enumerate(zip(file.readlines(), range(10))):
                js: PRMRecord = json.loads(line)

                samples: Iterable[Sample] = make_telescoping_conversation(js)
                batch = Batch([], [], [])
                instructions, response_lists, next_responses = batch
                try:
                    for sample_ix, sample in enumerate(samples):
                        instruction, responses, next_response = sample
                        instructions.append(instruction)
                        response_lists.append(responses)
                        next_responses.append(next_response)
                except GiveUp as e:
                    logger.warning(f'Record at line {line_ix} gave up, at conversation step {sample_ix+1}')
                
                if not batch.instructions:
                    continue

                table = pa.Table.from_arrays(list(batch), schema=schema)
                writer.write_table(table)