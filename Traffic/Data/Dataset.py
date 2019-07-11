"""
.. module:: DataSet

DataSet
*************

:Description: DataSet

    Object that obtains a set of images and transforms it into a matrix

:Authors: bejar
    

:Version: 

:Created on: 09/07/2019 13:51 

"""


import os.path

import h5py
import numpy as np
from numpy.random import shuffle
from collections import Counter

from Traffic.Util.Misc import list_days_generator, name_days_file
from Traffic.Config import Config

__author__ = 'bejar'

class Dataset:

    config = None

    def __init__(self, ldays, zfactor):
        """
        Checks if the file exists

        :param datapath:
        :param days:
        :param zfactor:
        :param nclases:
        :param merge: Merge classes
        """

        self.config = Config()
        self.fname = f'{self.config.datapath}/Data-{name_days_file(ldays)}-Z{zfactor:0.2f}.hdf5'
        self.X_train = None
        self.input_shape = None

        if not os.path.isfile(self.fname):
            raise Exception('Data file does not exists')
        self.handle = None

    def open(self):
        """
        Opens the hdf5 file
        :return:
        """
        self.handle = h5py.File(self.fname, 'r')

    def close(self):
        """
        Closes the hdf5 file
        :return:
        """
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            # Freeing memory
            del self.X_train

    def load_data(self):
        """
        Loads the data
        :return:
        """
        self.X_train = self.handle['data'][()]
        self.input_shape = self.X_train.shape[1:]


if __name__ == '__main__':

    from Traffic.Util.Misc import list_range_days_generator
    days = list_range_days_generator('20161201', '20161201')
    data = Dataset(days,0.25)

    data.open()
    data.load_data()

    print(data.input_shape)

    data.close()


