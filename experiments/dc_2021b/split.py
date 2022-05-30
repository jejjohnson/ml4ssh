from sklearn.model_selection import train_test_split

def add_split_args(parser):
    parser.add_argument("--train-size", type=float, default=0.90)
    parser.add_argument("--split-seed", type=int, default=666)
    return parser

def split_data(df, args):
    
    idx_train, idx_valid = train_test_split(
        df.index.values, train_size=args.train_size, random_state=args.split_seed)

    
    df["split"] = 0
    df.loc[idx_train, "split"] = 1
    df.loc[idx_valid, "split"] = 2
    
    cols = ["time_transform", "longitude", "latitude"]
    var_cols = ["sla_unfiltered"] 

    xtrain = df[df["split"].isin([1])][cols].values
    ytrain = df[df["split"].isin([1])][var_cols].values
    xvalid = df[df["split"].isin([2])][cols].values
    yvalid = df[df["split"].isin([2])][var_cols].values
    
    
    return xtrain, ytrain, xvalid, yvalid
