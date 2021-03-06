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
from Traffic.Models import WGAN, WGAN2
import warnings

warnings.filterwarnings("ignore")

__author__ = 'bejar'

models = {'WGAN': WGAN, 'WGAN2':WGAN2}

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
    parser.add_argument('--gkernel', default=3, type=int, help='Size of the convolutional kernel for the generator')
    parser.add_argument('--dkernel', default=3, type=int, help='Size of the convolutional kernel for the discriminator')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout probability')
    parser.add_argument('--nfilters', nargs='+', default=[128, 64], type=int, help='Number of convolutional filters')
    parser.add_argument('--dense', default=1024, type=int, help='Size of the dense layer')
    parser.add_argument('--nsamples', default=4, type=int, help='SQRT of the number of samples to generate')
    parser.add_argument('--resize', default=2, type=int, help='Number of image resizes performed by the generator')
    parser.add_argument('--saveint', default=10, type=int, help='Save samples every n epochs')
    parser.add_argument('--noisedim', default=100, type=int, help='Number of dimensions of the noise for the generator')
    parser.add_argument('--model', default='WGAN', type=str, help='Model used for training')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parser.parse_args()
    z_factor = float(args.zoom)

    days = list_range_days_generator(args.idate, args.fdate)
    data = Dataset(days, args.zoom)
    data.open()
    data.load_data()
    X_train = data.X_train
    data.close()

    MODEL = models[args.model]

    wgan = MODEL(image_dim=X_train.shape[1:], tr_ratio=args.trratio, gen_noise_dim=args.noisedim, num_filters=args.nfilters,
                gkernel=args.gkernel, dkernel=args.dkernel, nsamples=args.nsamples, dropout=args.dropout,
                exp=f'{args.idate}-{args.fdate}-Z{args.zoom}')

    wgan.train(X_train, args.epochs, batch_size=args.batch, sample_interval=args.saveint, verbose=args.verbose)

    print('Done.')
