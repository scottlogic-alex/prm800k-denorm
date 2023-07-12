from os import makedirs
from os.path import dirname, realpath
from pathlib import Path
import pyarrow as pa
from pyarrow.parquet import ParquetWriter

schema = pa.schema([
    ('instruction', pa.string()),
    ('responses', pa.list_(pa.string())),
    ('next_response', pa.string()),
])

if __name__ == '__main__':
    script_dir = Path(dirname(realpath(__file__)))
    repo_root: Path = script_dir.parent
    out_dir = repo_root.joinpath('out')
    makedirs(out_dir, exist_ok=True)
    with ParquetWriter(str(out_dir.joinpath('prm800k.parquet')), schema=schema) as writer:
        for _ in range(4):
            table = pa.Table.from_arrays([
                ['test'],
                [['test0', 'test1']],
                ['test'],
            ], names=['instruction', 'responses', 'next_response'])
            writer.write_table(table)