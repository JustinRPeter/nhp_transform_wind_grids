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
        self.direction = direction

    def process(self,data_map,msg):
        (v,data), = data_map.items()

        chunk = msg['content']['chunk']['in']

        z0 = self.z0[slice(*chunk['y']),slice(*chunk['x'])]
        zd = 0.5

        if self.direction == 'upward':
            ### factor to convert 2m to 10m
            k = np.log((10 - zd)/z0) / np.log((2 - zd)/z0)
        elif self.direction == 'downward':
            ### factor to convert 10m to 2m
            k =  np.log((2 - zd)/z0) / np.log((10 - zd)/z0)

        odata = k * data
        return processors.R(odata)



@elapsed(logger,success)
def transform_downwards(period, **kwargs):

    extent = get_default_extent()

    opath = kwargs['out_path']
    makedirs(opath,exist_ok=True)

    var_name = 'wswd' #'sfcWind'
    ipath = kwargs['in_path']
    var_map = {
        var_name: join(
            ipath,
            'wswd_' + ('[0-9]'*4) + '.nc')}

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
    mgr.num_consumers = 3
    mgr.num_readers = 4

    mgr.ichk_map = {var_name: (366,128,128)}
    mgr.ochk_map = {'out': (366,128,128)}

    mgr.setup(var_map,odb)
    status = mgr.run()

    if not status:
        raise Exception("PROCESS FAILED")



if __name__ == '__main__':
    args = dict(
        in_path='./prepared_files',
        out_path='./transform_grids_output',
        h5_grids='./davenport-vertical-wind-profile-parameters-0.05-mean.h5'
    )
    period = pd.date_range("1 jan 1960","31 dec 2005", freq='D')

    transform_downwards(period,**args)
