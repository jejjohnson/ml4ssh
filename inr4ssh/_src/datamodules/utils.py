from ..datasets.utils import ToTensor, TimeJulian, TimeMinMax, TimeJulianMinMax


def transform_factory(config):
    transforms = []

    if config.time_transform == "julian":
        transforms.append(TimeJulian())
    elif config.time_transform == "julian_minmax":
        transforms.append(
            TimeJulianMinMax(time_min=config.time_min, time_max=config.time_max)
        )
    elif config.time_transform == "minmax":
        transforms.append(
            TimeMinMax(time_min=config.time_min, time_max=config.time_max)
        )
    else:
        raise ValueError(f"Unrecognized transform: {config.time_transform}")

    # add the to tensor transformation
    transforms.append(ToTensor())

    return transforms


def get_evalulation_data(config):

    # create spatiotemporal grid
    lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
        lon_min=config.lon_min,
        lon_max=config.lon_max,
        lon_dx=config.dlon,
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lat_dy=config.dlat,
        time_min=np.datetime64(config.time_min),
        time_max=np.datetime64(config.time_max),
        time_dt=np.timedelta64(config.dtime_freq, config.dtime_unit),
    )

    df_grid = pd.DataFrame(
        {
            "longitude": lon_coords,
            "latitude": lat_coords,
            "time": time_coords,
        }
    )

    return df_grid
