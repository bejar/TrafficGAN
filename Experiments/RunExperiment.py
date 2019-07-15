"""
.. module:: RunExperiment

RunExperiment
*************

:Description: RunExperiment

    

:Authors: bejar
    

:Version: 

:Created on: 09/07/2019 13:21 

"""

import argparse
import h5py
import os

from Traffic.Util.TransformImages import generate_dataset
from Traffic.Util.Misc import list_range_days_generator, name_days_file
from Traffic.Config import Config
from Traffic.Data.Dataset import Dataset
from Traffic.Models.WGAN import WGAN
import warnings
warnings.filterwarnings("ignore")

__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--zoom', default=0.25, type=float, help='Zoom Factor')
    # parser.add_argument('--chunk', default='1024', help='Chunk size')
    # --test flag for including all data in the HDF5 file (last chunk is not the same size than the rest)
    # parser.add_argument('--test', action='store_true', default=False, help='Data generated for test')
    parser.add_argument('--idate', default='20161101', help='First day')
    parser.add_argument('--fdate', default='20161130', help='Final day')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs to train')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose output')
    parser.add_argument('--batch', default=64, type=int, help='Batch size')
    parser.add_argument('--trratio', default=5, type=int, help='Training Ratio')
    parser.add_argument('--nfilters', nargs='+', default=[128,64], type=int, help='Number of convolutional filters')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parser.parse_args()
    z_factor = float(args.zoom)


    days = list_range_days_generator(args.idate, args.fdate)
    data = Dataset(days, args.zoom)
    data.open()
    data.load_data()
    X_train = data.X_train
    data.close()

    wgan = WGAN(batch=args.batch, tr_ratio=args.trratio, num_filters=nfilters)

    wgan.train(X_train, args.epochs, args.verbose)

    print('Done.')

