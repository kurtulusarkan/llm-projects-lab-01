import argparse

from lab_01.compare import comparison_rows, format_table, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed YAML experiments.")
    parser.add_argument("path_or_prefix", nargs="?", help="Experiment directory or output-name prefix.")
    parser.add_argument("--csv", metavar="PATH", help="Write the comparison table as CSV.")
    args = parser.parse_args()

    rows = comparison_rows(path_or_prefix=args.path_or_prefix)
    print(format_table(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"csv_output: {args.csv}")


if __name__ == "__main__":
    main()
