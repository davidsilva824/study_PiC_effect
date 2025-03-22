# %%
from collections import Counter
from functools import partial
import argparse
import hashlib
import typing

from matplotlib import pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import surprisal


def cheap_hash(thing: str, n=6):
    return hashlib.md5(thing.encode("utf-8")).hexdigest()[:n].upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="model id to use for extracting surprisals. this may either be the huggingface model ID or an OpenAI model ID",
    )
    parser.add_argument(
        "-c",
        "--model_class",
        type=str,
        required=False,
        default=None,
        choices=["bert", "gpt", "gpt3"],
        help="model type/class. 'bert' and 'gpt' are two broad model classes for huggingface models. 'gpt3' is for gpt3-like models accessed using the OpenAI API",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")
    parser.add_argument(
        "--prefix",
        default="How likely is this: ",
        type=str,
        help="string to use as prefix for each AN pair to obtain surprisal",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="string to use as suffix for each AN pair to obtain surprisal",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to run only a small number of samples for testing purposes",
    )

    args = parser.parse_args()

    prefix = args.prefix
    suffix = args.suffix

    model = surprisal.AutoHuggingFaceModel.from_pretrained(
        args.model_name_or_path, model_class=args.model_class
    )

    df = pd.read_csv("vecchi2016_an_data_cogsci/annotations.csv")[
        ["unit_id", "which_makes_more_sense", "an1", "an2"]
    ]
    # df.head(4)

    all_pairs = df.an1.to_list() + df.an2.to_list()
    all_df = pd.DataFrame({"an": list(sorted(set(all_pairs)))})

    if args.debug:
        all_df = all_df.iloc[:128].copy()

    surprisals = []
    for an in tqdm(all_df.an.iloc[:]):
        a, n = an.split(" ")
        # NOTE: crucially, we include here the 'a' as part of the prefix so that
        # only the surprisal for the 'n' is extracted
        fn = partial(model.extract_surprisal, prefix=prefix + a + " ", suffix=suffix)
        surprisals += [fn(n)]

    all_df["prefix"] = prefix
    all_df["suffix"] = suffix
    all_df[args.model_name_or_path.replace("/", "-")] = list(
        map(lambda x: float(x[0]), surprisals)
    )
    all_df.to_csv(
        f"{args.output_dir}/vecchi2016_n_surprisals_{args.model_name_or_path.replace('/', '-')}_{cheap_hash(prefix+suffix)}.csv"
    )


if __name__ == "__main__":
    main()
