import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_feature_args(parser):
    parser.add_argument("--julian-time", type=bool, default=True)
    parser.add_argument("--feature-scaler", type=str, default="minmax")
    return parser

def feature_transform(df, args, scaler=None):

    # transform to julian time
    if args.julian_time == True:
        df["time_transform"] = pd.DatetimeIndex(df['time']).to_julian_date()
    else:
        df["time_transform"] = pd.DatetimeIndex(df['time'])
    
    # column transformer
    cols = ["time_transform", "longitude", "latitude"]
    if scaler is not None:
        df[cols] = scaler.transform(df[cols].values)

        return df
    elif args.feature_scaler == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(df[cols].values)
        df[cols] = scaler.transform(df[cols].values)
    else:
        scaler = None
        
    return df, scaler
