#
# Matt Dalle Piagge, Dec 2019
#-----------------------------------------
#
# Utilities for working with netcdf, grd and shapefile data
# Adapted July 2025 for the UKCEH Summer School

import numpy as np
import netCDF4 as nc4
import xarray as xr
import datetime as dt
from dateutil.relativedelta import relativedelta
import shapefile
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
import pandas as pd
import geopandas as gpd
from rasterio import features
from affine import Affine
import s3fs

# ------------------
# functions
def read_grd_file(filen, mask=0):
    vals = np.loadtxt(filen,skiprows=6)
    if mask==1:
        vals = np.ma.masked_values(vals, 0)
        vals = np.ma.masked_values(vals, -9999.)
        # preserve the mask whilst setting masked values to nans
        vals.data[np.where(vals.data == 0)]      = np.nan 
        vals.data[np.where(vals.data == -9999.)] = np.nan
    return vals

def print_array_size(data):
    print(np.shape(data))
    print('min ',np.nanmin(data))
    print('max ',np.nanmax(data))
    
def read_nc(filen,var):
    ncf = nc4.Dataset(filen,'r')    # open and read from netcdf file
    data = ncf.variables[var][:]    # variable
    x = ncf.variables['x'][:]       # easting
    y = ncf.variables['y'][:]       # northing
    if len(data.shape) >= 3:     # time
        t = ncf.variables['t'][:]
        tunits = ncf.variables['t'].units
    else:
        t = None
        tunits = None
    ncf.close()                     # close netcdf file  
    return data,x,y,t,tunits
    
def get_coords(filein):
    filen = open(filein,'r')
    ncols = int(filen.readline().split()[1])
    nrows = int(filen.readline().split()[1])
    xllc  = float(filen.readline().split()[1])
    yllc  = float(filen.readline().split()[1])
    res   = float(filen.readline().split()[1])
    nodata= float(filen.readline().split()[1])
    filen.close()
    return nrows, ncols, xllc, yllc, res, nodata

def make_xarray(filein):
    nrows, ncols, xllc, yllc, res, nodata = get_coords(filein)
    xcoords = np.linspace(xllc + (res/2), xllc + (res/2) + (res*(ncols-1)), ncols)
    ycoords = np.linspace(yllc + (res/2), yllc + (res/2) +  (res*(nrows-1)), nrows)[::-1]
    vals = np.loadtxt(filein, skiprows=6)
    vals = xr.DataArray(vals, coords=[('y', ycoords), ('x', xcoords)])
    vals = vals.where(vals != nodata)
    return vals, xcoords, ycoords, nodata

def nctoxr(filein, var):
    data,x,y = read_nc(filein, var)
    dataxr = xr.DataArray(data, coords=[('y',y), ('x',x)])
    return dataxr

def subset_to_template(filein, template):
    tvals, xcoords_template, ycoords_template, nodata = make_xarray(template)
    tvalsnp = tvals.values.copy()
    vals, xcoords, ycoords, nodata = make_xarray(filein)
    subvals = vals.sel(y = slice(ycoords_template[0], ycoords_template[-1]),
                       x  = slice(xcoords_template[0], xcoords_template[-1]))
    newxcoords = subvals.coords['x'].values
    newycoords = subvals.coords['y'].values
    return subvals, newxcoords, newycoords, nodata

def read_csv_orig(direc, years, suffix, prefix, usecodes):

    years = list(years)
    usecodes = list(usecodes)
    file1 = direc + str(years[0]) + prefix + suffix
    datatemp = np.loadtxt(file1)
    gridpoints = datatemp[:,:2]
    ngridpoints = datatemp.shape[0]
    gridpointinds = []
    # use np.unique to find out how many actual gridpoints are (as those with multiple
    # usecodes will be duplicated).
    # need to format these in such a way that each lat and each lon is the same length
    # 0-pad basically. Then can easily split them to obtain the lat and lon.
    # Also always start with a 1 so that the initial leading zeros don't get chopped off.
    # We need to have them as one number originally for np.unique to work.
    for xy in range(0, ngridpoints):
        gridpointinds.append(int('1{:010.0f}{:010.0f}'.format(datatemp[xy,0], datatemp[xy,1])))
    gpia = np.asarray(gridpointinds)
    gpiau = list(np.unique(gpia))
    gpiaus = [str(x) for x in gpiau]
    nugp = len(gpiaus)

    nfiles = len(years)

    # Create array
    data = np.zeros((nugp, len(usecodes), len(years), 12))

    # Loop over lines, note the gridpoints, assign values to array
    counter = 0
    for year in years:
        counter += 1
        print('Processing file ' + str(counter) + ' of ' + str(nfiles))
        filein = direc + str(year) + prefix + suffix

        yearidx = years.index(year)

        tempdata = np.loadtxt(filein) 

        for row in range(0,tempdata.shape[0]):
            gridpoint = '1{:010.0f}{:010.0f}'.format(tempdata[row,0], tempdata[row,1])
            usecode = int(tempdata[row,2])
            gpidx = gpiaus.index(gridpoint)
            ucidx = usecodes.index(usecode)

            data[gpidx, ucidx, yearidx, :] += tempdata[row,3:]

    # have just 1 time dimension
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))

    # format the gridpoint coords in a more user friendly way: (easting, northing)
    # the index corresponds to that of the data array. So coords[0] is the
    # coordinates of data[0,:,:,:]
    coords = [(int(x[1:11]),int(x[11:])) for x in gpiaus]

    return data, coords

def int2ordinal(num):
    def formatted_num(suffix):   # Nested function.
        return str(num) + suffix

    end_digits = int(str(num)[-1:])   # Gets last char of the string. Instead of repeating this operation on every if clause, better do it here.
    # You only really need to check if it either finishes with 1,2,3 - remaining values will have a th suffix.
    if end_digits in [1]:
        return formatted_num('st')
    elif end_digits in [2]:
        return formatted_num('nd')
    elif end_digits in [3]:
        return formatted_num('rd')
    else:
        return formatted_num('th')

def grd_to_netcdf_single(filein, varname, units, filesave=1, filenameout='grd.ncf', subset=0, template=None):
    if subset == 1:
        vals, junk1, junk2, nodata = subset_to_template(filein, template)
    else:
        vals, junk, junk2, nodata = make_xarray(filein)
    vals.name = varname
    vals.attrs = {'Units': units, 'missing_value': nodata}
    vals.coords['x'].attrs = {'long_name':'easting', 'standard_name':'projection_x_coordinate', 'units':'m', 'point_spacing':'even', 'axis':'x'}
    vals.coords['y'].attrs = {'long_name':'northing', 'standard_name':'projection_y_coordinate', 'units':'m', 'point_spacing':'even', 'axis':'y'}

    if filesave == 1:
        vals.to_netcdf(filenameout)
    else:
        return vals, nodata
    
def grds_to_netcdf(files, year, varname, units, usecodes=None, aggregate=0, filesave=1, filenameout='grds.ncf', subset=0, template=None):
    '''
    Function to convert grd files to netcdf, one year at a time. If files are split by usecode, the function
    currently aggregates these. Assumes the list of files is sorted by usecode, i.e. all
    the 010 usecode files, then all the 020 usecode files... And that each date has the 
    same number of files/usecodes. 

    files: List of lists of files, each sublist containing the files corresponding to varname
    varname: List of variable names corresponding to each sublist of files
    units: List of unit names corresponding to the varnames
    startyear: Year the files start from. Assumes files are monthly and that they start in Jan of startyear
    usecodes: List of usecodes which should be present in the filenames (typically 3digit 0padded integers)
    filesave: 0 to output xarray Dataset or 1 to save file as netcdf. 1 by default.
    filenameout: Filename of output netcdf file. Ignored if filesave=0. 
    subset: Subset down to a template grid (1) or not (0). 0 by default.
    template: Filename of grd file that acts as the template grid to subset to. Ignored if subset=0.
    '''

    # work out dimensions of the grd files
    if subset == 1:
        temp, xcoords, ycoords, nodata = subset_to_template(files[0], template)
    else:
        temp, xcoords, ycoords, nodata = make_xarray(files[0])


    # work out dimensions needed for the array that will store all the data from all the files
    timedimlen = 12
    ydimlen = temp.values.shape[0]
    xdimlen = temp.values.shape[1]

    if usecodes:
        allvals = np.zeros((ydimlen, xdimlen, timedimlen, len(usecodes)))
    else:
        allvals = np.zeros((ydimlen, xdimlen, timedimlen))    

        
    # Create the time units for the nc file later
    startdate = dt.datetime(year, 1, 1)
    tdates = [startdate + relativedelta(months=i) for i in range(0, timedimlen)]
    tunits = 'days since ' + str(year) + '-01-01'
    tcoords = nc4.date2num(tdates, tunits)


    # calc total number of files
    totalfiles = len(files)

    # read data into main array
    varind = 0
    counter = 0
    timedimcounter = -1
    fileind = 0

    for filein in files:
        counter+=1
        print('Reading in file ' + str(counter) + ' of ' + str(totalfiles) + ': ' + filein)

        # pull out data from individual file
        if subset == 1:
            vals, junk, junk2, junk3 = subset_to_template(filein, template)
            vals = vals.values
        else:
            vals = read_grd_file(filein)

        # assign to main array
        if usecodes:
            # if the files are split by usecode then this is a little more tricky
            # the files cycle through all the usecodes for a certain date, before
            # moving on to the next date. Therefore, the time dimension index should
            # only increment every multiple of the total number of usecodes, and the
            # usecode dimension index should reset to zero at the same time.
            # The '%' line below means find the remainder when you divide the first
            # by the second. It's zero when the filenumber is a multiple of the
            # total number of usecodes (including zero). 
            rem = fileind%len(usecodes)
            if rem == 0:
                timedimcounter += 1                    
            allvals[:,:,timedimcounter,rem] = vals
        else:
            allvals[:,:,fileind] = vals

        fileind+=1

    # intermediate save 
    #np.save('allvals', allvals)
    
    # sum over usecodes
    if usecodes:
        if aggregate == 1:
            allvals = np.nansum(allvals, axis = 3)

    # create netcdf dataset using xarray
    if usecodes:
        save_as_ncf(allvals, xcoords, ycoords, tcoords, tunits, varname, units, nodata, filenameout, usecodes=usecodes)
    else:
        save_as_ncf(allvals, xcoords, ycoords, tcoords, tunits, varname, units, nodata, filenameout)


def save_as_ncf(data, xcoords, ycoords, tcoords, tunits, varname, units, nodata, filenameout, usecodes = None):

    allvars = xr.Dataset()
    
    if not usecodes:
        usecodes = ['']
        data = np.expand_dims(data, axis = -1)

    for uc in range(0, len(usecodes)):
            allvalsxr = xr.DataArray(data[:,:,:,uc], coords=[('y', ycoords), ('x', xcoords), ('t', tcoords)])
            allvalsxr = allvalsxr.where(allvalsxr != nodata)    
            allvalsxr.name = usecodes[uc] + varname
            allvalsxr.attrs = {'Units': units, 'missing_value': nodata, '_FillValue': nodata}
            allvalsxr.coords['x'].attrs['long_name'] = 'easting'
            allvalsxr.coords['y'].attrs['long_name'] = 'northing'
            allvalsxr.coords['x'].attrs['standard_name'] = 'projection_x_coordinate'
            allvalsxr.coords['y'].attrs['standard_name'] = 'projection_y_coordinate'
            allvalsxr.coords['x'].attrs['units'] = 'm'
            allvalsxr.coords['y'].attrs['units'] = 'm'
            allvalsxr.coords['x'].attrs['point_spacing'] = 'even'
            allvalsxr.coords['y'].attrs['point_spacing'] = 'even'
            allvalsxr.coords['x'].attrs['axis'] = 'x'
            allvalsxr.coords['y'].attrs['axis'] = 'y'
            allvalsxr.coords['t'].attrs['units'] = tunits
            allvalsxr.coords['t'].attrs['calendar'] = 'standard'
            allvars = allvars.assign({usecodes[uc] + varname: allvalsxr})
        
    print('Saving netcdf file to: ' + filenameout)
    allvars.to_netcdf(filenameout)

def concat_vars(dataset, dimname, dimvals, varname):
    alldata = xr.concat([dataset[key] for key in dataset.data_vars], dim=dimname)
    alldata.name = varname
    alldata[dimname] = (dimname, dimvals)
    return alldata


def catchment_subset(data, template, cIDs):
    '''
    Function to only select out the datapoints of an xarray dataset
    corresponding to a particular catchment/river basin area defined by a grd
    text file
    '''

    # Read in template
    print('Reading in template file: ' + template)
    template, junk, junk2, junk3 = make_xarray(template)

    # Subset
    print('Subsetting to catchments ' + str(cIDs))
    subset = data.where(template.isin(cIDs))

    return subset


def catchment_subset_pointdata(data, coords, sfname, IDname, IDs):
    '''
    Function to subset ungridded point timeseries data to selected shapes from
    a shapefile.
    data: Array. 2D or 3D. Typically [gridpoint, time] or [gridpoint, usecode, time]
    coords: A list of tuples containing the (easting, northing) coordinates of each gridpoint
            len(coords) should match data.shape[0] 
    sfname: The filename of the shapefile
    IDname: The name of the catgeory to search over for selecting shapes to subset to (e.g. 'River')
    IDs:    The values of the category to select (e.g. ['Thames', 'Severn'])
    '''

    # Read in shapefile
    srs = []
    sf = shapefile.Reader(sfname)

    # Select out the shapes of interest
    print('Finding catchments...')
    for ID in IDs:
        for shaperecord in sf.shapeRecords():
            if shaperecord.record.as_dict()[IDname] == ID:
                srs.append(shaperecord)
                print(shaperecord.record)

    # Get coords of the shape polygons
    polygons = []
    shapecoords = [0 for x in srs]
    for shapen in range(0, len(srs)):
        shapecoords[shapen] = srs[shapen].shape.points
        polygons.append(Polygon(srs[shapen].shape.points))

    # work out which coords are within the shapefiles of interest
    newdata = np.zeros(data.shape)
    newcoords = []
    shapecounter = 0
    dataindcounter = 0
    for poly in polygons:
        shapecounter += 1
        print('Processing catchment ' + str(shapecounter) + ' of ' + str(len(polygons)))
        inside_points = np.array(MultiPoint(coords).intersection(poly))
        for cor in inside_points:
            cor = tuple(cor)
            if cor not in newcoords:
                newcoords.append(cor)
                cordidx = coords.index(cor)
                newdata[dataindcounter] = data[cordidx]
                dataindcounter += 1

    newdata = newdata[:dataindcounter]

    return newdata, newcoords, srs

def catchment_subset_shapefile(data=None, datafile=None, multifile=0, sfname=None, endpoint=None, xname='x', yname='y', IDname=None, IDs=None, drop=0):
    '''
    Function to subset an xarray dataarray or dataset, or netcdf dataset, to selected shapes from 
    a shapefile. Returns an xarray dataset with of the same shape as the input
    datafile but with the data outside the selected shapes
    set to nans. Also returns the shapes so these can be plotted.

    data:     An xarray DataArray or DataSet
    datafile: The filename of the netcdf file to subset. Multiple files can be selected with * etc.
              If this is the case multifile should be set to 1. Defaults to 0.
    sfname:   The filepath of the shapefile
    endpoint: Used if sfname is an s3 path. The endpoint_url for the s3 storage system being used. 
    IDname:   The name of the catgeory to search over for selecting shapes to subset to (e.g. 'RIVER')
    IDs:      The values of the category to select (e.g. ['Thames', 'Severn'])
    multifile:Are multiple files specfied in datafile? Set to 1 if so. In this case the files are read in
              using dask, which can process data that exceeds the memory capacity of the machine by
              processing in parallel.
    xname: Name of the x-coordinate in the netcdf file(s). 'x' by default.
    yname: Name of the y-coordinate in the netcdf file(s). 'y' by default
    '''

    # Read in data
    if datafile:
        print('Reading in ' + datafile)
        if multifile == 1:
            data = xr.open_mfdataset(datafile, parallel=True)
        else:
            data = xr.load_dataset(filein)

    subset = add_shape_coord_from_data_array(data, sfname, endpoint, IDname, IDs, yname, xname)
    if drop == 0:
        subset = subset.where(subset[IDname]==1, other=np.nan)
    else:
        subset = subset.where(subset[IDname]==1, drop=True)
        
        
    return subset



def add_shape_coord_from_data_array(xr_da, shp_path, endpoint, IDname, IDs, latname, lonname):
    """ Create a new coord for the xr_da indicating whether or not it 
    is inside the shapefile
    
    Creates a new coord - "coord_name" which will have integer values
    used to subset xr_da for plotting / analysis/
    
    Usage:
    -----
    precip_da = add_shape_coord_from_data_array(precip_da, "awash.shp", "awash")
    awash_da = precip_da.where(precip_da.awash==0, other=np.nan) 
    """
    # 1. read in shapefile
    if shp_path[:5] == "s3://":
        s3 = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint)
        shp_gpd = gpd.read_file(s3.open(shp_path))
    else:
        shp_gpd = gpd.read_file(shp_path)
    
    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = []
    counter = 0
    for ID in shp_gpd[IDname]:
        if ID in IDs:
            shapes.append((shp_gpd['geometry'][counter], 1))
            print('Found: ' + str(shp_gpd[IDname][counter]))
        counter+=1

    if len(shapes) == 0:
        raise AttributeError(IDname + ' ' + str(IDs) + ' not found in shapefile')

    xr_da[IDname] = rasterize(shapes, xr_da.coords,
                              longitude=lonname, latitude=latname)
    
        
    return xr_da




def transform_from_latlon(lat, lon):
    """ 
    input 1D array of lat / lon and output an Affine transformation
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='y', longitude='x',
              fill=np.nan, **kwargs):
    """
    Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.
    """
                  
    print('Adding mask to xarray')
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))
        
def replace_t_axis(data, groupby):
    # replaces one of the existing axes of an xarray dataarray with
    # an identical one but with a datetime type rather than integer
    # or something, and with the name 't'
    # Useful when using the groupby xarray function, which removes
    # datetime functionality from the time axis, and only tested
    # in this scenario when grouping by year
    newname = data.name
    newdata = data.copy()
    newdata = newdata[groupby].reindex({groupby: pd.to_datetime([str(time) for time in list(data[groupby].values)])})
    newdata.values = data.squeeze().values
    newdata = newdata.rename({groupby: 't'})
    newdata = newdata.rename(newname)

    return newdata
