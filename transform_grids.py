import h5py
import math
import numpy as np
from os import makedirs
from os.path import join
import argparse
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
    DIRECTION_UP = 'upward'
    DIRECTION_DOWN = 'downward'

    def __init__(self, h5_file, direction):
        h = h5py.File(h5_file,'r')
        self.z0 = h['parameters/z0_masked'][:]
        self.direction = direction

    def process(self,data_map,msg):
        (v,data), = data_map.items()

        chunk = msg['content']['chunk']['in']

        z0 = self.z0[slice(*chunk['y']),slice(*chunk['x'])]
        zd = 0.5

        if self.direction == TransformWind.DIRECTION_UP:
            ### factor to convert 2m to 10m
            k = np.log((10 - zd)/z0) / np.log((2 - zd)/z0)
        elif self.direction == TransformWind.DIRECTION_DOWN:
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
    variable.attrs['wind_transformed'] = '10m to 2m' if direction == TransformWind.DIRECTION_DOWN else '2m to 10m'
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



def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_glob_pattern', '-i', required=True,
        help='Input file glob pattern')
    parser.add_argument(
        '--output_path', '-o', required=True,
        help='Output path. Folder will be created if it doesn\'t already exist')
    parser.add_argument(
        '--grid_file', '-g', required=True,
        help='Grid file containing z0 factor. Expected to be in H5 format, with parameter name parameters/z0_masked')
    parser.add_argument(
        '--year_start', '-ys', required=True, type=int,
        help='Start Year. 1st Jan of start year is assumed')
    parser.add_argument(
        '--year_end', '-ye', required=True, type=int,
        help='End Year. 31st Dec of end year is assumed')
    parser.add_argument(
        '--wind_var_name', '-n', required=True,
        help='Wind variable name')
    parser.add_argument(
        '--direction', '-d', required=True,
        choices=[TransformWind.DIRECTION_UP, TransformWind.DIRECTION_DOWN],
        help='Direction to transform. upward = 2m to 10m, downward = 10m to 2m')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = handle_arguments()
    print('Transform Grids run with:')
    print('Input glob pattern:', args.input_glob_pattern)
    print('Output path:', args.output_path)
    print('Grid file:', args.grid_file)
    print('Wind variable name:', args.wind_var_name)
    print('Start year:', args.year_start)
    print('End year:', args.year_end)
    print('Direction:', args.direction)

    period = pd.date_range(
        '{:d}-01-01'.format(args.year_start),
        '{:d}-12-31'.format(args.year_end), freq='D')

    transform(
        input_glob_pattern = args.input_glob_pattern,
        out_path = args.output_path,
        h5_grids_file = args.grid_file,
        wind_var_name = args.wind_var_name,
        period = period,
        direction = args.direction)
