"""
.. module:: Generate_Dataset

Generate_Dataset
*************

:Description: Generate_Dataset

    Different functions for Traffic dataset generation

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 12:53 

"""

import glob
from collections import Counter
import numpy as np
from Traffic.Config.Cameras import Cameras_ok
from Traffic.Config.Constants import cameras_path, dataset_path, process_path, status_path
from Traffic.Data.DataTram import DataTram
from Traffic.Process.CamTram import CamTram
import pickle
import h5py
from Traffic.Data.TrafficImage import TrImage
from Traffic.Util.Misc import list_days_generator, name_days_file, dist_time, get_hour

__author__ = 'bejar'


def info_dataset(path, ldaysTr, z_factor, imgordering='th'):
    """
    Prints counts of the labels of the dataset for a list of days

    :param path:
    :param imgordering:
    :param ldaysTr:
    :param z_factor:
    :return:
    """

    y_train = []
    for day in ldaysTr:
        data = np.load(path + 'labels-D%s-Z%0.2f.npy' % (day, z_factor))
        X = np.load(path + 'data-D%s-Z%0.2f-%s.npy' % (day, z_factor, imgordering))
        print(day, Counter(data), data.shape[0], X.shape)
        y_train.extend(data)
    print('TOTAL=', Counter(list(y_train)), len(y_train))


def get_day_images_data(day, cpatt=None):
    """
    Return a dictionary with all the camera identifiers that exist for all the timestamps of the day
    cpatt allows to select only some cameras that match the pattern
    :param day:
    :param cpatt:
    :return:
    """
    if cpatt is not None:
        ldir = sorted(glob.glob(cameras_path + day + '/*' + cpatt + '*.gif'))
    else:
        ldir = sorted(glob.glob(cameras_path + day + '/*.gif'))

    camdic = {}

    for f in sorted(ldir):
        name = f.split('.')[0].split('/')[-1]
        time, place = name.split('-')
        if place in Cameras_ok:
            if int(time) in camdic:
                camdic[int(time)].append(place)
            else:
                camdic[int(time)] = [place]

    return camdic


def get_day_predictions(day):
    """
    Returns all the predictions for a day

    :param day:
    :return:
    """
    ldir = glob.glob(status_path + day + '/*-dadestram.data')
    if not ldir:
        raise Exception('Day does not exists')
    ldata = []
    for f in sorted(ldir):
        ldata.append(DataTram(f))
    return ldata


def generate_classification_dataset_one(day, cpatt=None):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status using only the nearest prediction in space and time

    Version ONE

    (superseded by generate_labeled_dataset_day)

    :param cpatt:
    :param day:
    :return:
    """
    camdic = get_day_images_data(day, cpatt=cpatt)
    ldata = get_day_predictions(day)
    CTram = CamTram()
    assoc = {}

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        dmin2 = None
        vmin2 = 70
        vmin = 60
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff):  # Only if it is ahead in time
                if imgtime - d.date > 0:
                    vmin2 = vmin
                    vmin = diff
                    dmin2 = dmin
                    dmin = d
            elif vmin2 > np.abs(diff):
                if diff > 0 and diff != vmin:
                    vmin2 = diff
                    dmin2 = d

        if dmin is not None and dmin2 is not None and vmin < 60 and vmin2 < 60:
            lclass = []
            for img in camdic[imgtime]:
                tram = CTram.ct[img][0]
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img, dmin.dt[tram][0], dmin.dt[tram][1], dmin2.dt[tram][0]))
            assoc[imgtime] = lclass

    return assoc


def generate_classification_dataset_two(day, cpatt=None, mxdelay=60):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status using the two nearest prediction in space and time

    Version TWO

    (superseded by generate_labeled_dataset_day)

    :param cpatt:
    :param day:
    :param mxdelay: Maximum delay distance between image and status label
    :return:
    """

    camdic = get_day_images_data(day, cpatt=cpatt)
    ldata = get_day_predictions(day)
    assoc = {}
    CTram = CamTram()

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        dmin2 = None
        vmin = 60
        vmin2 = 70
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff):  # Only if it is ahead in time
                if diff >= 0:
                    vmin2 = vmin
                    vmin = diff
                    dmin2 = dmin
                    dmin = d
            elif vmin2 > np.abs(diff):
                if diff >= 0 and diff != vmin:
                    vmin2 = diff
                    dmin2 = d
        if dmin is not None and dmin2 is not None and vmin < mxdelay and vmin2 < mxdelay:
            print(vmin, vmin2)
            lclass = []
            for img in camdic[imgtime]:
                tram1 = CTram.ct[img][0]
                tram2 = CTram.ct[img][1]
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img,
                               max(dmin.dt[tram1][0], dmin.dt[tram2][0]),
                               max(dmin.dt[tram1][1], dmin.dt[tram2][1]),
                               max(dmin2.dt[tram1][0], dmin2.dt[tram2][0])))
            assoc[imgtime] = lclass

    return assoc


def generate_dataset(ldaysTr, z_factor, method='one', cpatt=None):
    """
    Generates a training and test datasets from the days in the parameters
    z_factor is the zoom factor to rescale the images
    :param cpatt:
    :param ldaysTr:
    :param z_factor:
    :param method:
    :return:

    """
    ldata = []
    llabels = []
    limages = []
    image = TrImage()
    for day in ldaysTr:
        if method == 'one':
            dataset = generate_classification_dataset_one(day, cpatt=cpatt)
        else:
            dataset = generate_classification_dataset_two(day, cpatt=cpatt)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                if l != 0 and l != 6:

                    image.load_image(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if image.is_correct():
                        ldata.append(image.transform_image(z_factor=z_factor, crop=(5, 5, 5, 5)))
                        llabels.append(l)
                        limages.append(day + '/' + str(t) + '-' + cam)

    print(Counter(llabels))

    X_train = np.array(ldata)
    y_train = llabels

    return X_train, y_train, limages


def generate_data_day(day, z_factor, method='two', mxdelay=60, log=False):
    """
    Generates a raw dataset for a day with a zoom factor (data, labels and list of image files)

    Superseded by generate_labeled_dataset_day

    :param log:
    :param mxdelay:
    :param method:
    :param day:
    :param z_factor:
    :return:
    """
    ldata = []
    llabels = []
    limages = []
    if method == 'one':
        dataset = generate_classification_dataset_one(day)
    else:
        dataset = generate_classification_dataset_two(day, mxdelay=mxdelay)
    for t in dataset:
        for cam, l, _, _ in dataset[t]:
            if l != 0 and l != 6:
                if log:
                    print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')

                image = TrImage(cameras_path + day + '/' + str(t) + '-' + cam + '.gif', z_factor=z_factor,
                                crop=(5, 5, 5, 5))
                if image.correct():
                    ldata.append(image.get_data())
                    llabels.append(l)
                    limages.append(day + '/' + str(t) + '-' + cam)

    X_train = np.array(ldata)
    X_train = X_train.transpose((0, 3, 1, 2))  # Theano ordering
    llabels = [i - 1 for i in llabels]  # change labels from 1-5 to 0-4
    np.save(dataset_path + 'data-D%s-Z%0.2f.npy' % (day, z_factor), X_train)
    np.save(dataset_path + 'labels-D%s-Z%0.2f.npy' % (day, z_factor), np.array(llabels))
    output = open(dataset_path + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'wb')
    pickle.dump(limages, output)
    output.close()


# --------------------------------------------------------------------------------------
# New functions for generating the datasets

def generate_image_labels(day, mxdelay=30, onlyfuture=True, hourban=(None,None)):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current
    traffic status using the two nearest prediction in space

    :param onlyfuture:
    :param day:
    :param mxdelay: Maximum delay distance between image and status label
    :return:
    """

    camdic = get_day_images_data(day)
    ldata = get_day_predictions(day)
    assoc = {}
    CTram = CamTram()

    for imgtime in sorted(camdic):
        if hourban[0] is not None and hourban[1] is not None and hourban[0] <= get_hour(imgtime) <= hourban[1]:
            continue
        if hourban[0] is not None and hourban[1] is None and hourban[0] <= get_hour(imgtime) <= 24:
            continue
        if hourban[0] is None and hourban[1] is not None and 0 <= get_hour(imgtime) <= hourban[1]:
            continue

        # Look for the status and forecast closer to the image only future or future and past
        dmin = None
        vmin = 100

        # Find the closest prediction in time for the day
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff):
                if onlyfuture:
                    if diff >= 0:
                        vmin = np.abs(diff)
                        dmin = d
                else:
                    vmin = np.abs(diff)
                    dmin = d

        if dmin is not None and vmin < mxdelay:
            # print vmin, imgtime, dmin.date
            lclass = []
            for img in camdic[imgtime]:
                # Two closest positions to the camera (stored in file CameraTram.txt)
                tram1 = CTram.ct[img][0]
                tram2 = CTram.ct[img][1]

                # store for an image of that time the name and worst status from the two closest positions
                lclass.append((img, max(dmin.dt[tram1][0], dmin.dt[tram2][0])))
            assoc[imgtime] = lclass
    return assoc


def generate_labeled_dataset_day(path, day, z_factor, mxdelay=60, onlyfuture=True, log=False, imgordering='th',
                                 augmentation=[], hourban=(None,None), badpixels=[None,None]):
    """
    Generates a raw dataset for a day with a zoom factor (data and labels)
    :param augmentation:
    :param imgordering:
    :param log:
    :param onlyfuture:
    :param mxdelay:
    :param day:
    :param path:
    :param z_factor:
    :return:
    """
    ldata = []
    llabels = []
    limages = []

    dataset = generate_image_labels(day, mxdelay=mxdelay, onlyfuture=onlyfuture, hourban=hourban)
    image = TrImage()
    for t in dataset:
        for cam, l in dataset[t]:
            if l != 0 and l != 6:
                if log:
                    print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                image.load_image(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if image.is_correct():
                    image.transform_image(z_factor=z_factor, crop=(5, 5, 5, 5))
                    if badpixels[0] is None or not image.bad_pixels(badpixels[0], badpixels[1]):
                        ldata.append(image.get_data())
                        llabels.append(l)
                        limages.append(day + '/' + str(t) + '-' + cam)
                        if l - 1 in augmentation:
                            aug = image.data_augmentation()
                            for cnt, im in enumerate(aug):
                                ldata.append(im)
                                llabels.append(l)
                                limages.append(day + '/' + str(t) + '-' + cam + str(cnt))

    X_train = np.array(ldata)
    if imgordering == 'th':
        X_train = X_train.transpose((0, 3, 1, 2))  # Theano image ordering

    llabels = [i - 1 for i in llabels]  # change labels from 1-5 to 0-4
    np.save(path + 'data-D%s-Z%0.2f-%s.npy' % (day, z_factor, imgordering), X_train)
    np.save(path + 'labels-D%s-Z%0.2f.npy' % (day, z_factor), np.array(llabels))
    output = open(path + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'wb')
    pickle.dump(limages, output)
    output.close()


def load_generated_day(datapath, day, z_factor, imgordering='th'):
    """
    Load the dataset for a given day returns a data matrix, a list of labels an a list
    of the names of the files of the examples

    :param imgordering:
    :param day:
    :param datapath:
    :param z_factor:
    :return:
    """

    X_train = np.load(datapath + 'data-D%s-Z%0.2f-%s.npy' % (day, z_factor, imgordering))
    y_train = np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor))
    output = open(datapath + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'rb')
    img_path = pickle.load(output)
    output.close()

    return X_train, y_train, img_path


def chunkify(lchunks, size, test=False):
    """
    Returns the saving list for the data with chunks_list of size = size
    :param test:
    :param lchunks:
    :param size:
    :return:
    """

    accum = 0
    csize = size
    i = 0
    quant = lchunks[0]
    lcut = []
    lpos = []

    while i < len(lchunks):
        if accum + quant <= size:
            accum += quant
            lpos.append((i, quant))
            i += 1
            if i < len(lchunks):
                quant = lchunks[i]
            else:
                if accum == size or test:
                    lcut.append(lpos)
        else:
            lpos.append((i, size - accum))
            lcut.append(lpos)
            lpos = []
            quant -= (size - accum)
            accum = 0
            csize += size

    return lcut


def generate_training_dataset(datapath, ldays, chunk=1024, z_factor=0.25, imgordering='th', test=False,
                              compress='gzip', augmentation=[]):
    """
    Generates an hdf5 file for a list of days with blocks of data for training
    It need the files for each day, the data is grouped and chunked in same sized
    datasets

    Some data may be discarded is the total number of examples is not a multiple of chunk

    :param compress:
    :param test:
    :param imgordering:
    :param z_factor:
    :param chunk:
    :param datapath:
    :param ldays:
    :return:
    """

    nlabels = []
    for i, day in enumerate(ldays):
        labels = np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor))
        nlabels.append(len(labels))

    lsave = chunkify(nlabels, chunk, test=test)

    nf = name_days_file(ldays)
    sfile = h5py.File(datapath + '/Data-%s-Z%0.2f-%s.hdf5' % (nf, z_factor, imgordering), 'w')

    prev = {}
    for nchunk, save in enumerate(lsave):
        curr = {}
        for nday, nex in save:
            curr[ldays[nday]] = [load_generated_day(datapath, ldays[nday], z_factor, imgordering), nex, 0]
            if ldays[nday] in prev:
                curr[ldays[nday]][2] += prev[ldays[nday]][1]

        X_train = []
        y_train = []
        imgpath = []
        for day in sorted(curr):
            indi = int(curr[day][2])
            indf = int(curr[day][2] + curr[day][1])

            X_train.append(curr[day][0][0][indi:indf])
            y_train.extend(curr[day][0][1][indi:indf])
            imgpath.extend(curr[day][0][2][indi:indf])

        X_train = np.concatenate(X_train)
        y_train = np.array(y_train, dtype='d')
        imgpath = [n.encode("ascii", "ignore") for n in imgpath]
        prev = curr

        namechunk = 'chunk%03d' % nchunk
        sfile.require_dataset(namechunk + '/' + 'data', X_train.shape, dtype='f',
                              data=X_train, compression=compress)

        sfile.require_dataset(namechunk + '/' + 'labels', y_train.shape, dtype='i',
                              data=y_train, compression=compress)

        sfile.require_dataset(namechunk + '/' + 'imgpath', (len(imgpath), 1), dtype='S100',
                              data=imgpath, compression=compress)
        sfile.flush()
    sfile.close()


if __name__ == '__main__':
    # days = list_days_generator(2016, 11, 1, 30) + list_days_generator(2016, 12, 1, 3)
    days = list_days_generator(2016, 12, 1, 2)
    z_factor = 0.25

    # Old day datafiles generation
    # for day in days:
    #     generate_data_day(day, z_factor, method='two', mxdelay=60)

    # Uncomment to view information of day datafiles (examples per class)
    # info_dataset(process_path, days, z_factor, imgordering='tf')

    # Uncomment to generate files for a list of days
    # for day in days:
    #     generate_labeled_dataset_day(process_path, day, z_factor, mxdelay=15, onlyfuture=False, imgordering='th')

    # Uncoment to generate a HDF5 file for a list of days
    generate_training_dataset(process_path, days, chunk=256, z_factor=z_factor, imgordering='th')
