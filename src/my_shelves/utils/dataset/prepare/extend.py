import os

import pandas as pd

import my_shelves.utils.dataset.prepare.fake_locations as fake_locations


N_ROWS_MAP = {"10k": 10_000,
              "100k": 100_000,
              "1M": 1_000_000,
              # "10M": 10_000_000,
              "all": None
}


def extend_dataset_with_fake_locations(df):
    """
    Étend le dataset existant avec des lieux touristiques fictifs.

    Parameters
    ----------
    df : pd.DataFrame
        Le dataset existant à étendre.
    n_rows : int
        Le nombre de lignes fictives à ajouter.

    Returns
    -------
    pd.DataFrame
        Le dataset étendu avec les lieux fictifs.
    """
    n_rows = df.shape[0]
    fake_gen = fake_locations.FakeLocations()
    fake_df = fake_gen.generate(n=n_rows)
    print(fake_df.head())
    # Concaténer le dataset existant avec le dataset fictif
    extended_df = pd.concat([df, fake_df], axis=1, join="inner")

    return extended_df


if __name__ == "__main__":
    for lang in ["ENG"]:
        # for nrows in N_ROWS_MAP.keys():
        for nrows in["100k"]:
            input_file = f"data/reviews_cleaned_{lang}_{nrows}.csv"
            output_file = f"data/reviews_extended_{lang}_{nrows}.csv"

            # if os.path.exists(output_file):
            #     print(f"{output_file} already exists. Skipping extension.")
            #     continue

            print("_" * 80)
            print(f"Extending dataset for {lang} with {nrows} rows...")
            print("_" * 80)

            print(f"Reading input file {input_file}...")
            df = pd.read_csv(input_file)
            print("extending dataset with fake locations...")
            extended_df = extend_dataset_with_fake_locations(df)
            print(f"Saving extended dataset to {output_file}...")
            extended_df.to_csv(output_file, index=False)
            print(f"Extended dataset saved to {output_file}.")
            print(extended_df.head())
