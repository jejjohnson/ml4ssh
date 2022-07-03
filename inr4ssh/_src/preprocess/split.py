from sklearn.model_selection import train_test_split


def split_data(df, train_size, split_seed):

    idx_train, idx_valid = train_test_split(
        df.index.values, train_size=train_size, random_state=split_seed
    )

    df["split"] = 0
    df.loc[idx_train, "split"] = 1
    df.loc[idx_valid, "split"] = 2

    cols = df.attrs["input_cols"]
    var_cols = df.attrs["output_cols"]

    xtrain = df[df["split"].isin([1])][cols].values
    ytrain = df[df["split"].isin([1])][var_cols].values
    xvalid = df[df["split"].isin([2])][cols].values
    yvalid = df[df["split"].isin([2])][var_cols].values

    return xtrain, ytrain, xvalid, yvalid
