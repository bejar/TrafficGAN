"""
.. module:: GenerateData

GenerateData
*************

:Description: GenerateData

    

:Authors: bejar
    

:Version: 

:Created on: 20/02/2017 8:26 

"""
import argparse
from Traffic.Util.TransformImages import generate_dataset
from Traffic.Util.Misc import list_range_days_generator, name_days_file
from Traffic.Config import Config
import h5py

__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--zoom', default='0.25', help='Zoom Factor')
    # parser.add_argument('--chunk', default='1024', help='Chunk size')
    # --test flag for including all data in the HDF5 file (last chunk is not the same size than the rest)
    # parser.add_argument('--test', action='store_true', default=False, help='Data generated for test')
    parser.add_argument('--idate', default='20161101', help='First day')
    parser.add_argument('--fdate', default='20161130', help='Final day')
    # Dicards images from an hour larger or equal than the first element and  less or equal than the second element
    # parser.add_argument('--hourban', nargs='+', default=[None, None], help='Initial ban hour', type=int)
    #
    # parser.add_argument('--augmentation', nargs='+', default=[], help='Use data augmentation for certain classes',
    #                     type=int)
    # The badpixels filted discards the images using a histogram of colors, it has two values
    # number of bins for the histogram and a percentage that is the threshold for discarding the image
    parser.add_argument('--greyhisto', nargs='+', default=None,
                        help='Apply the greyscale histogram filter to the images', type=int)
    # number of bins for the histogram and a percentage that is the threshold for discarding the image
    parser.add_argument('--lapent', nargs='+', default=None,
                        help='Apply the combined laplacian-entropy filter to the images', type=int)
    # for compresssing uses as value in the parameter gzip
    parser.add_argument('--compress', action='store_true', default=False, help='Compression for the HDF5 file')
    # parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the images')

    args = parser.parse_args()

    z_factor = float(args.zoom)
    # imgord = args.imgord
    # chunk = int(args.chunk)
    # mxdelay = int(args.delay)
    days = list_range_days_generator(args.idate, args.fdate)
    compress = 'gzip' if args.compress else None

    # if len(args.hourban) > 0:
    #     if args.hourban[0] is None or 0 <= args.hourban[0] <= 23:
    #         ihour = args.hourban[0]
    #     else:
    #         raise Exception('Parameters for HOURBAN incorrect hours in [0,23]')

    # if len(args.hourban) == 2:
    #     if args.hourban[0] is None or 0 <= args.hourban[1] <= 23:
    #         fhour = args.hourban[1]
    #     else:
    #         raise Exception('Parameters for HOURBAN incorrect hours in [0,23]')
    # else:
    #     fhour = None

    if args.greyhisto is not None:
        if len(args.greyhisto) == 2:
            nbin = args.greyhisto[0]
            if 0 < args.greyhisto[1] <= 100:
                perc = args.greyhisto[1]
            else:
                raise Exception('Parameters for GREYHISTO incorrect Percentage in (0,100]')
        else:
            raise Exception('Parameters for GREYHISTO incorrect')
    else:
        nbin, perc = None, None

    if args.lapent is not None:
        if len(args.lapent) == 4:
            lap1, lap2, bins, ent = args.lapent
        else:
            raise Exception('Parameters for LAPLACIAN-ENTROPY incorrect')
    else:
        lap1, lap2, bins, ent = None, None, None, None

    print('Generating data for:')
    print('IDAY = ', days[0])
    print('FDAY = ', days[-1])
    print('ZOOM_FACTOR = ', z_factor)
    # print 'IMAGE_ORDER = ', imgord
    # print 'CHUNK_SIZE = ', chunk
    # print 'MAX_DELAY = ', mxdelay
    # print 'TEST = ', args.test
    # print 'SHUFFLE = ', args.shuffle
    # if ihour is not None:
    #     print 'I_HOUR_BAN = ', args.hourban[0]
    # if fhour is not None:
    #     print 'F_HOUR_BAN = ', args.hourban[1]
    if args.greyhisto is not [None, None]:
        print('GREY_HISTO = ', args.greyhisto)
    if args.lapent is not [None, None, None, None]:
        print('LAPLACIAN_ENTROPY = ', args.lapent)
    # if args.augmentation:
    #     print 'AUGMENTATION = ', args.augmentation
    print('COMPRESS = ', compress)

    print()
    print('Processing images ...')
    print()
    data = generate_dataset(days, z_factor=z_factor,
                     badpixels=(nbin, perc), lapent=(lap1, lap2, bins, ent))

    print(f'{data.shape[0]} Images')

    config = Config()
    nf = name_days_file(days)
    sfile = h5py.File(f'{config.datapath}/Data-{nf}-Z{z_factor:0.2f}.hdf5', 'w')

    sfile.require_dataset('data', data.shape, dtype='f',
                              data=data, compression=compress)
    sfile.flush()
    sfile.close()