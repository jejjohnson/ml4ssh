import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def add_feature_args(parser):
    parser.add_argument("--julian-time", type=bool, default=True)
    parser.add_argument("--feature-scaler", type=str, default="minmax")
    return parser

def feature_transform(df, args, scaler=None):

    # transform to julian time
    if args.julian_time == True:
        df["time"] = pd.DatetimeIndex(df['time']).to_julian_date().copy()
    else:
        df["time"] = df["vtime"].copy()
    
    # column transformer
    cols = df.attrs["input_cols"]
    if scaler is not None:
        print(cols)
        df[cols] = scaler.transform(df[cols].values.copy())

        return df
    elif args.feature_scaler == "none":
        scaler = None
    elif args.feature_scaler == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(df[cols].values.copy())
        df[cols] = scaler.transform(df[cols].values.copy())
        
    elif args.feature_scaler == "standard":
        scaler = StandardScaler().fit(df[cols].values.copy())
        df[cols] = scaler.transform(df[cols].values.copy())
    else:
        scaler = None
        
    return df, scaler
