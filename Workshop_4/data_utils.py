import polars as pl



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


def make_region_hght_clip(region_shp, hght_data):
    """Clip region by height raster"""
    region_clip = hght_data.rio.clip(
        region_shp.geometry.values, region_shp.crs, drop=False, invert=False
    )
    return region_clip


def binarize_hght_clip(region_data):
    """
    Make binary mask of clipped region data data
    """
    return region_data / region_data.where(region_data > 0)


def mask_region_rainfall(rainfall_data, region_mask):
    """
    Mask region by binary height raster
    """
    return rainfall_data * binarize_hght_clip(region_mask).data


def mask_region_rainfall_by_hght(rainfall_data, region_hght, threshold):
    """
    Mask region by height raster
    """
    region_hght_mask = region_hght / region_hght.where(region_hght > threshold)
    return rainfall_data * region_hght_mask.data
