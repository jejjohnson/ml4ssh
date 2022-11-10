import pandas as pd
import numpy as np


# def get_evalulation_data(config):

#     # create spatiotemporal grid
#     lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
#         lon_min=config.lon_min,
#         lon_max=config.lon_max,
#         lon_dx=config.dlon,
#         lat_min=config.lat_min,
#         lat_max=config.lat_max,
#         lat_dy=config.dlat,
#         time_min=np.datetime64(config.time_min),
#         time_max=np.datetime64(config.time_max),
#         time_dt=np.timedelta64(config.dtime_freq, config.dtime_unit),
#     )

#     df_grid = pd.DataFrame(
#         {
#             "longitude": lon_coords,
#             "latitude": lat_coords,
#             "time": time_coords,
#         }
#     )

#     return df_grid
