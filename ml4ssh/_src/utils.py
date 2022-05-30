import numpy as np
import pyinterp
import xarray as xr

def temporal_subset(ds, time_min, time_max, time_buffer):
    ds = ds.sel(
        time=slice(time_min - np.timedelta64(int(2*time_buffer), 'D'), 
                   time_max + np.timedelta64(int(2*time_buffer), 'D')),
        drop=True
    )
    return ds

def spatial_subset(ds, lon_min, lon_max, lon_buffer, lat_min, lat_max, lat_buffer):
    
    # correct lon if domain is between [-180:180]
    if lon_min < 0:
        ds['longitude'] = xr.where(
            ds['longitude'] >= 180., 
            ds['longitude']-360., 
            ds['longitude']
        )
        
    ds = ds.where(
        (ds['longitude'] >= lon_min - lon_buffer) & 
        (ds['longitude'] <= lon_max + lon_buffer) &
        (ds['latitude'] >= lat_min - lat_buffer) &
        (ds['latitude'] <= lat_max + lat_buffer) , 
        drop=True
    )
    
    return ds

def get_meshgrid(res: float, nx: int, ny: int):
    dx = res
    dy = res
    x = np.linspace(-1, 1, int(nx)) * (nx - 1) * dx / 2
    y = np.linspace(-1, 1, int(ny)) * (ny - 1) * dy / 2
    return np.meshgrid(x, y)


def calculate_gradient(da, x_coord: str="Nx", y_coord: str="Ny", edge_order=2):
    
    # first marginal derivative
    dx = da.differentiate(coord=x_coord, edge_order=2)
    dy = da.differentiate(coord=y_coord, edge_order=2)

    return 0.5 * (dx**2 + dy**2)

def calculate_laplacian(da, x_coord: str="Nx", y_coord: str="Ny", edge_order=2):
    
    # second marginal derivative
    dx2 = da.differentiate(coord=x_coord, edge_order=2).differentiate(coord=x_coord, edge_order=2)
    dy2 = da.differentiate(coord=x_coord, edge_order=2).differentiate(coord=x_coord, edge_order=2)

    return 0.5 * (dx2**2 + dy2**2)


def create_spatiotemporal_coords(
    lon_min, lon_max, lon_dx,
    lat_min, lat_max, lat_dy,
    time_min, time_max, time_dt):
    
    # create all coordinates
    glon = np.arange(lon_min, lon_max + lon_dx, lon_dx)           # output OI longitude grid
    glat = np.arange(lat_min, lat_max + lat_dy, lat_dy)           # output OI latitude grid
    gtime = np.arange(time_min, time_max + time_dt, time_dt)        # output OI time grid
    
    return glon, glat, gtime

def create_spatiotemporal_grid(
    lon_min, lon_max, lon_dx,
    lat_min, lat_max, lat_dy,
    time_min, time_max, time_dt):
    
    glon, glat, gtime = create_spatiotemporal_coords(
        lon_min, lon_max, lon_dx,
        lat_min, lat_max, lat_dy,
        time_min, time_max, time_dt
    )        # output OI time grid
    
    # Make 3D grid
    glon2, glat2, gtime2 = np.meshgrid(glon, glat, gtime)
    lon_coords = glon2.flatten()
    lat_coords = glat2.flatten()
    time_coords = gtime2.flatten()

    return lon_coords, lat_coords, time_coords


def get_gridded_data(ds: xr.Dataset, 
                    lon_min=0., 
                    lon_max=360., 
                    lat_min=-90, 
                    lat_max=90., 
                    time_min='1900-10-01', 
                    time_max='2100-01-01', 
                    is_circle=True):
    
    
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["longitude"]%360. >= lon_min) & (ds["longitude"]%360. <= lon_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)
    
    x_axis = pyinterp.Axis(ds["longitude"][:].values%360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["latitude"][:].values)
    z_axis = pyinterp.TemporalAxis(ds["time"][:].values)
    
    var = ds['ssh'][:]
    var = var.transpose('longitude', 'latitude', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass
    
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid

def create_grids(ds: xr.DataArray, variable: str, is_circle=True):
    x_axis = pyinterp.Axis(ds["longitude"][:]%360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["latitude"][:])
    z_axis = pyinterp.TemporalAxis(ds["time"][:].values)
    var = ds[variable][:]
    var = var.transpose("longitude", "latitude", "time")
    
    try:
        var[var.mask] = np.nan
    except AttributeError:
        pass
    
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.values)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid

def rename_coords(ds):
    return None