from pandas import read_parquet, DataFrame
from os.path import dirname, realpath
from pathlib import Path

if __name__ == '__main__':
    script_dir = Path(dirname(realpath(__file__)))
    repo_root: Path = script_dir.parent
    out_dir = repo_root.joinpath('out')

    # df: DataFrame = read_parquet(str(out_dir.joinpath('phase1_train.parquet')))
    # df: DataFrame = read_parquet(str(out_dir.joinpath('phase1_train.answer_only.parquet')))
    df: DataFrame = read_parquet(str(out_dir.joinpath('phase1_train.critique.parquet')))
    pass # somewhere to put a breakpoint
