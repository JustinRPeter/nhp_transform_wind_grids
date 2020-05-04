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
def transform(
        input_glob_pattern,
        out_path,
        h5_grids_file,
        wind_var_name,
        period,
        direction):

    extent = get_default_extent()

    makedirs(out_path, exist_ok=True)

    var_map = {wind_var_name:input_glob_pattern}

    idb = DBManager(var_map[wind_var_name], var_name=wind_var_name)
    variable = idb.variable
    variable.pars['chunksizes'] = (32,32,32)
    variable.dtype = np.dtype('float32')
    variable.attrs._FillValue = np.float32(-999.)
    variable.dimensions = ['time','latitude','longitude']
    odb = DBManager.create_annual_split_from_extent(out_path, variable, period, extent)

    processor = TransformWind(h5_grids_file, direction=direction)
    processor.handle_period(period)

    mgr = TaskManager(processor,extent)
    mgr.num_consumers = 3
    mgr.num_readers = 4

    mgr.ichk_map = {wind_var_name: (366,128,128)}
    mgr.ochk_map = {'out': (366,128,128)}

    mgr.setup(var_map, odb)
    status = mgr.run()

    if not status:
        raise Exception("PROCESS FAILED")



if __name__ == '__main__':
    in_path='./prepared_files'
    input_glob_pattern = join(in_path, 'wswd_' + ('[0-9]'*4) + '.nc')
    period = pd.date_range("1 jan 1960","31 dec 2005", freq='D')

    transform(
        input_glob_pattern = input_glob_pattern,
        out_path = './transform_grids_output',
        h5_grids_file = './davenport-vertical-wind-profile-parameters-0.05-mean.h5',
        wind_var_name = 'wswd',
        period = period,
        direction = 'downward'):
