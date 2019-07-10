"""
.. module:: TransformImages

TransformImages
*************

:Description: TransformImages

    

:Authors: bejar
    

:Version: 

:Created on: 10/07/2019 13:21 

"""

import glob

from Traffic.Cameras import Cameras_ok
from Traffic.Util.Misc import list_days_generator
from Traffic.Config import Config
from Traffic.Data.TrafficImage import TrafficImage

__author__ = 'bejar'


def generate_dataset(ldaysTr, z_factor, bad_pixels=[None,None], lapent=[None, None, None, None]):
    """
    Generates a training and test datasets from the days in the parameters
    z_factor is the zoom factor to rescale the images
    :param cpatt:
    :param ldaysTr:
    :param z_factor:
    :param method:
    :return:

    """

    config = Config()
    cameras_path = config.cameraspath

    ldata = []
    image = TrafficImage()
    for day in ldaysTr:
        camdic = get_day_images_data(config.cameraspath, day)
        for t in camdic:
            print(t, camdic[t])
            image.load_image(cameras_path + day + '/' + str(t) + '-' + camdic[t] + '.gif')
            if not image.corrupted and image.is_correct():
                image.transform_image(z_factor=z_factor, crop=(0, 0, 0, 0))
                if image.trans:
                    filter_image = False
                    if badpixels[0] is not None:
                        filter_image = filter_image or image.greyscale_histo(badpixels[0], badpixels[1])
                        # filter_image = filter_image or image.truncated_bad_pixels(badpixels[0], badpixels[1])
                    if lapent[0] is not None:
                        laplacian = image.var_laplacian()
                        # Filter if laplacian is less than first threshold
                        filter_image = filter_image or laplacian < lapent[0]
                        # if not filter if laplacian is less than second threshold and
                        # entropy is less than entropy threshold
                        if not filter_image and laplacian < lapent[2]:
                            entropy = image.entropy(lapent[3])
                            filter_image =  entropy < lapent[4]

                    if not filter_image:
                        ldata.append(image.get_data())
                        print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')

    X_train = np.array(ldata)

    return X_train

def get_day_images_data(path, day):
    """
    Return a dictionary with all the camera identifiers that exist for all the timestamps of the day
    cpatt allows to select only some cameras that match the pattern
    :param day:
    :param cpatt:
    :return:
    """

    ldir = sorted(glob.glob(path + day + '/*.gif'))

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

if __name__ == '__main__':
    # days = list_days_generator(2016, 11, 1, 30) + list_days_generator(2016, 12, 1, 3)
    days = list_days_generator(2016, 12, 1, 2)
    config = Config()


    data = generate_dataset(days,0.5)
    print(data.shape)

