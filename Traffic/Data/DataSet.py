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

__author__ = 'bejar'


class Dataset:
    def __init__(self, datapath, ldays, zfactor, augmentation=False):
        """
        Checks if the file exists

        :param datapath:
        :param days:
        :param zfactor:
        :param nclases:
        :param merge: Merge classes
        """
        aug = '-aug' if augmentation else ''
        self.fname = datapath + '/' + "Data-" + name_days_file(ldays) + '-Z%0.2f%s-%s' % (zfactor, aug, imgord) + '.hdf5'
        self.recode = recode
        self.labels_prop = None
        self.X_train = None
        self.y_labels = None
        self.input_shape = None
        self.chunk_size = None
        self.chunks = None
        self.y_train = None
        self.perm = None

        if not os.path.isfile(self.fname):
            raise Exception('Data file does not exists')
        self.handle = None

        # If the dataset is going to be loaded in batches we need the number of classes
        if nclasses is not None:
            self.nclasses = nclasses