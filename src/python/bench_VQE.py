import argparse
import VQE
import polars as pl
from typing import Callable
from pathlib import Path

from config import BENCH_DIR

def bench_ham(func: Callable[[], VQE.VQEResult], rounds: int, name: str) -> pl.DataFrame:
    res_list = []
    for _ in range(rounds):
        iter_res = func()
        res_list.append(iter_res)
    df = pl.DataFrame(res_list)
    avg_df = df.mean()
    name_df = pl.DataFrame({"name": name})
    combined_df = pl.concat([name_df, avg_df], how="horizontal")
    return combined_df

parser = argparse.ArgumentParser()
parser.add_argument("out_file")
args = parser.parse_args()
out_path = Path(args.out_file)
out_path.parent.mkdir(exist_ok=True, parents=True)

reps=5
spin = bench_ham(VQE.RunSpinHam, reps, "spin_cuda_2x4_isotropic")
n2 = bench_ham(VQE.RunN2Ham, reps, "N2_cuda_sto-3g_R-HF")
nh3 = bench_ham(VQE.RunNH3Ham, reps, "NH3_cuda_sto-3g_R-HF_is_frozen")

final_df = pl.concat([spin, n2, nh3])
print(final_df)
final_df.write_csv(out_path)
