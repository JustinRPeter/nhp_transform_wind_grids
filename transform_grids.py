import h5py
import math
import numpy as np
from os import makedirs
from os.path import join
import pandas as pd
import scipy.signal

from awrams.utils.extents import get_default_extent
from processing.data_interface import DBManager
from processing.support.managers import TaskManager
import processing.support.processors as processors
from job.postprocessing.util import elapsed, set_local_logging

import logging
logger = logging.getLogger(__file__)
logger = set_local_logging(logger)
success = "Finished"


class TransformWind(processors.Processor):
    def __init__(self, h5_file, direction='upward'):
        h = h5py.File(h5_file,'r')
        self.z0 = h['parameters/z0_masked'][:]
        # self.zd = h['parameters/zd'][:]
        self.direction = direction

    def process(self,data_map,msg):
        (v,data), = data_map.items()

        chunk = msg['content']['chunk']['in']

        z0 = self.z0[slice(*chunk['y']),slice(*chunk['x'])]
        zd = 0.5 #self.zd[slice(*chunk['y']),slice(*chunk['x'])]

        if self.direction == 'upward':
            ### factor to convert 2m to 10m
            k = np.log((10 - zd)/z0) / np.log((2 - zd)/z0)
        elif self.direction == 'downward':
            ### factor to convert 10m to 2m
            k =  np.log((2 - zd)/z0) / np.log((10 - zd)/z0)

        odata = k * data
        return processors.R(odata)

def extrap_nearest(mask):
    xx, yy = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    xx = xx.astype(np.int32)
    yy = yy.astype(np.int32)
    xym = np.vstack((np.ravel(xx[~mask.mask]), np.ravel(yy[~mask.mask]))).T

    # the valid values as 1D array (in the same order as coordinates in xym)
    data = np.ravel(mask.data[~mask.mask])

    # interpolator
    interp = scipy.interpolate.NearestNDInterpolator(xym, data)
    result = interp(np.ravel(xx).astype(np.int32), np.ravel(yy).astype(np.int32)).reshape(xx.shape)
    return result

class FilterGrids(processors.Processor):
    def handle_period(self,period,chunk_length=32):
        self.operiod = []
        self.iperiod = []
        self.max_len = 0

        for y in np.unique(period.year):
            p = period[period.year == y]
            idx = [chunk_length * (i+1) for i in range(math.floor(len(p) / chunk_length))]
            p_chunked = np.array_split(p,idx)
            # print(p_chunked)
            self.operiod.extend(p_chunked)
        self.max_len = chunk_length

        self.iperiod = self.operiod

    def process(self,data_map,msg):
        (v,data), = data_map.items()

        h = scipy.signal.hamming(25)
        h2d = np.sqrt(np.outer(h,h))
        h2d /= h2d.sum()
        # h2d = np.tile(h2d, (data.shape[0],1,1))

        # print(data.shape,h2d.shape)
        odata = data.copy()
        for i in range(data.shape[0]):
            # odata = scipy.signal.convolve(data.filled(0),h2d,mode='same') #,boundary='symm')
            # odata[i,:] = scipy.signal.convolve2d(data[i,:].filled(0),h2d,mode='same',boundary='symm')
            odata[i,:] = scipy.signal.convolve2d(
                extrap_nearest(data[i,:]), h2d, mode='same', boundary='symm')

        return processors.R(odata)


@elapsed(logger,success)
def main(period, **kwargs):

    extent = get_default_extent()

    opath = kwargs['out_path']
    makedirs(opath,exist_ok=True)

    var_name = 'wind'
    ipath = kwargs['in_path']
    var_map = {var_name: join(ipath,var_name+"_[0-9][0-9][0-9][0-9].nc")}

    idb = DBManager(var_map[var_name])
    variable = idb.variable
    variable.pars['chunksizes'] = (32,32,32)
    variable.dtype = np.dtype('float32')
    variable.attrs._FillValue = np.float32(-999.)
    odb = DBManager.create_annual_split_from_extent(opath,variable,period,extent)

    processor = TransformWind(kwargs['h5_grids'], direction='downward')
    processor.handle_period(period)

    mgr = TaskManager(processor,extent)
    mgr.num_consumers = 3 #2 #1
    mgr.num_readers = 4 #2 #1

    mgr.ichk_map = {var_name: (366,128,128)}
    mgr.ochk_map = {'out': (366,128,128)}

    mgr.setup(var_map,odb)
    status = mgr.run()

    if not status:
        raise Exception("PROCESS FAILED")


@elapsed(logger,success)
def transform_downwards(period, **kwargs):

    extent = get_default_extent()

    opath = kwargs['out_path']
    makedirs(opath,exist_ok=True)

    var_name = 'sfcWind'
    ipath = kwargs['in_path']
    var_map = {
        var_name: join(
            ipath,
            var_name + "_day_CCAM-r3355-ACCESS1-0_historical_r1i1p1_[0-9][0-9][0-9][0-9].nc")}

    idb = DBManager(var_map[var_name], var_name=var_name)
    variable = idb.variable
    variable.pars['chunksizes'] = (32,32,32)
    variable.dtype = np.dtype('float32')
    variable.attrs._FillValue = np.float32(-999.)
    variable.dimensions = ['time','latitude','longitude']
    odb = DBManager.create_annual_split_from_extent(opath,variable,period,extent)

    processor = TransformWind(kwargs['h5_grids'], direction='downward')
    processor.handle_period(period)

    mgr = TaskManager(processor,extent)
    mgr.num_consumers = 3 #2 #1
    mgr.num_readers = 4 #2 #1

    mgr.ichk_map = {var_name: (366,128,128)}
    mgr.ochk_map = {'out': (366,128,128)}

    mgr.setup(var_map,odb)
    status = mgr.run()

    if not status:
        raise Exception("PROCESS FAILED")


@elapsed(logger,success)
def smooth(period,**kwargs):
    extent = get_default_extent()

    var_name = 'wind'
    ipath = kwargs['out_path']
    var_map = {var_name: join(ipath,var_name+"_[0-9][0-9][0-9][0-9].nc")}

    opath = join(ipath, 'filtered')
    makedirs(opath,exist_ok=True)

    idb = DBManager(var_map[var_name])
    variable = idb.variable
    variable.pars['chunksizes'] = (32,32,32)
    variable.dtype = np.dtype('float32')
    variable.attrs._FillValue = np.float32(-999.)
    odb = DBManager.create_annual_split_from_extent(opath,variable,period,extent,filename='wind_filtered')

    temporal_chunk = 16
    processor = FilterGrids()
    processor.handle_period(period,chunk_length=temporal_chunk)

    mgr = TaskManager(processor,extent)
    mgr.num_consumers = 8
    mgr.num_readers = 1

    mgr.ichk_map = {var_name: (temporal_chunk,681,841)}
    mgr.ochk_map = {'out': (temporal_chunk,681,841)}

    mgr.setup(var_map,odb)
    status = mgr.run()

    if not status:
        raise Exception("PROCESS FAILED")


if __name__ == '__main__':
    # args = dict(
    #     in_path='/data/cwd_awra_data/awra_inputs/climate_generated_sdcvd-awrap01/wind/',
    #     # out_path='/data/cwd_awra_data/AWRAMSI/IWRM_0042_WP3/GIT/Stuart/py3/wind/transform-2m-grids-to-10m/output/',
    #     out_path='/data/cwd_awra_data/awra_inputs/climate_generated_sdcvd-awrap01/wind/transformed-to-10m/',
    #     h5_grids='/data/cwd_awra_data/AWRAMSI/IWRM_0042_WP3/GIT/Stuart/py3/wind/transform-2m-grids-to-10m/davenport-vertical-wind-profile-parameters-0.05-mean.h5'
    # )
    # # period = pd.date_range("1 jan 2018","31 dec 2018", freq='D')
    # period = pd.date_range("1 jan 1976","31 dec 2018", freq='D')
    
    # main(period,**args)
    # # smooth(period,**args)

    args = dict(
        in_path='/data/from_nci_tp28',
        out_path='./transfrom_grids_output',
        h5_grids='./davenport-vertical-wind-profile-parameters-0.05-mean.h5'
    )
    period = pd.date_range("1 jan 1960","31 dec 2005", freq='D')

    transform_downwards(period,**args)
