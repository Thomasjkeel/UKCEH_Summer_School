import fsspec
import zarr

import polars as pl
import xarray as xr

# GLOBALS that have been set for convenience
fdri_fs = fsspec.filesystem("s3", asynchronous=True, anon=True, endpoint_url="https://fdri-o.s3-ext.jc.rl.ac.uk")
gear_daily_zstore = zarr.storage.FsspecStore(fdri_fs, path="geardaily/GB/geardaily_fulloutput_yearly_100km_chunks.zarr")
gear_daily = xr.open_zarr(gear_daily_zstore, decode_times=True, decode_cf=True)

CEH_GEAR_DATA = gear_daily
CEH_GEAR_DATA = CEH_GEAR_DATA.drop_vars(('crs', 'lat', 'lon'))
SEVERN_GAUGE_DATA = pl.read_csv('s3://rain-gauge/hourly_severn_rain_gauge_data.csv', storage_options={'endpoint_url': "https://fdri-o.s3-ext.jc.rl.ac.uk", 'anon': True}, try_parse_dates=True)
SEVERN_METADATA = pl.read_csv('s3://rain-gauge/hourly_severn_rain_gauge_metadata.csv', storage_options={'endpoint_url': "https://fdri-o.s3-ext.jc.rl.ac.uk", 'anon': True})

SEVERN_GAUGE_DATA = SEVERN_GAUGE_DATA.rename({"DATETIME": "time", "PRECIPITATION": "rain_mm"})

RAIN_COL = "rain_mm"
CEH_GEAR_RAIN_VAR = "rainfall_amount"

class Gauge:
    def __init__(self, gauge_id, nearby_threshold_m):
        self.gauge_id = int(gauge_id)
        self.gauge_metadata = self._get_gauge_metadata()
        self.gauge_easting = self.gauge_metadata['EASTING'].item()
        self.gauge_northing = self.gauge_metadata['NORTHING'].item()
        self.gauge_data = self._get_gauge_data()
        self.nearby_gauges = self._get_nearby_gauges(nearby_threshold_m)
        self.closest_cehgear = self._get_closest_gridded_data(CEH_GEAR_DATA, nearby_threshold_m)

    def _get_gauge_metadata(self):
        assert self.gauge_id in SEVERN_METADATA['ID'],\
            f"Gauge id={self.gauge_id} is not in the gauge metadata"
        return SEVERN_METADATA.filter(pl.col("ID") == self.gauge_id)

    def _get_gauge_data(self):
        assert self.gauge_id in SEVERN_GAUGE_DATA['ID'],\
            f"Gauge id={self.gauge_id} is not in the gauge data"
        return SEVERN_GAUGE_DATA.filter(pl.col("ID") == self.gauge_id)

    def _get_nearby_gauges(self, nearby_threshold_m):
        nearby_gauge_metadata = SEVERN_METADATA.filter((pl.col('EASTING') >= self.gauge_easting-nearby_threshold_m)\
                                                & (pl.col('EASTING') <= self.gauge_easting+nearby_threshold_m) &\
            (pl.col('NORTHING') >= self.gauge_northing-nearby_threshold_m) & (pl.col('NORTHING') <= self.gauge_northing+nearby_threshold_m))
        return nearby_gauge_metadata['ID'].to_list()

    def _get_closest_gridded_data(self, gridded_data, closeness_threshold_m):
        """TODO: needs testing with data that is off grid"""
        closest_grid_cell = gridded_data.sel(x=self.gauge_easting, y=self.gauge_northing, method='nearest')
        distance_x, distance_y = abs(self.gauge_easting-closest_grid_cell['x']), abs(self.gauge_northing-closest_grid_cell['y'])
        if distance_x > closeness_threshold_m or distance_y > closeness_threshold_m:
            raise ValueError(f"Closest data point in grid is more than {closeness_threshold_m} m away in X ({distance_x} m), or Y ({distance_y} m)")
        return closest_grid_cell

    def get_nearby_gridded_data(self, gridded_data, nearby_radius_m):
        nearby_grid_cells = gridded_data.sel(x=slice(self.gauge_easting-nearby_radius_m, self.gauge_easting+nearby_radius_m),\
                                y=slice(self.gauge_northing-nearby_radius_m, self.gauge_northing+nearby_radius_m))
        return nearby_grid_cells


def get_combined_gauge_data(gauge, how='left'):
    assert isinstance(gauge, Gauge), "data should be of Gauge type"
    closest_ceh = pl.from_pandas(gauge.closest_cehgear[CEH_GEAR_RAIN_VAR].drop_vars(('x', 'y')).to_dataframe(f'{RAIN_COL}_closest_ceh').reset_index())
    closest_ceh = convert_time_to_hour_base(closest_ceh, hour=9)
    combined_data = gauge.gauge_data[['time', f'{RAIN_COL}']].join(closest_ceh, on='time', how=how)
    combined_data = combined_data.sort(by='time')
    return combined_data


def get_combined_gauge_data_w_nearby(gauge, nearby_radius_m, how='left'):
    combined_data = get_combined_gauge_data(gauge, how=how)
    gauge.nearby_ceh = gauge.get_nearby_gridded_data(CEH_GEAR_DATA, nearby_radius_m)
    nearby_ceh = pl.from_pandas(gauge.nearby_ceh[CEH_GEAR_RAIN_VAR].mean(('x', 'y')).to_dataframe(f'{RAIN_COL}_nearby_ceh').reset_index())
    nearby_ceh = convert_time_to_hour_base(nearby_ceh, hour=9)
    combined_data = combined_data.join(nearby_ceh, on='time', how=how)
    combined_data = combined_data.sort(by='time')
    return combined_data


def convert_time_to_hour_base(data, hour, time_col="time"):
    """For polars dataframes to convert timestamp to 9 am.
    Both HadUK-Grid and CEH-GEAR are base 9am-9am, but the
    data is stored differently."""
    return data.with_columns(
        pl.datetime(
            pl.col(time_col).dt.year(),
            pl.col(time_col).dt.month(),
            pl.col(time_col).dt.day(),
            hour,
            0,
            0,
        ).alias(time_col)
    )
