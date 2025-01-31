import argparse
import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    df = pl.read_csv(args.in_file)
    # print(df)
    agg = df.group_by([pl.col("qubits"), pl.col("depth")], maintain_order=True).agg(
        pl.col("cpu").mean(),
        pl.col("gpu").mean(),
        pl.col("exit_cpu").max(),
        pl.col("exit_gpu").max(),
    )
    # print(agg)
    agg.write_csv(args.out_file)


if __name__ == "__main__":
    main()
