"""
.. module:: TrImage

TrImage
*************

:Description: TrImage

    Class to read and process Traffic Images.
    For now only the images from Barcelona are considered for checking no service image

    The process is always:

    Create the object once

    for each image
        load the image
        check if it is correct
        transform the image crop+zoom (the object returns the data)


:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 7:12 

"""

from scipy.ndimage import zoom#, imread
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
# from Traffic.Config.Constants import info_path
import glob
import cv2
from Traffic.Config import Config

__author__ = 'bejar'


class TrafficImage:
    def __init__(self):
        """
        Object to process camera images.
        It loads no services images

        :param pimage:
        """
        self.config = Config()
        self.bcnnoserv = np.asarray(Image.open(self.config.infopath + 'BCNnoservice.gif'))
        self.correct = False
        self.data = None
        self.trans = False
        self.fname = None
        self.corrupted = False
        self.normalized = False

    def load_image(self, pimage):
        """
        Loads an image from a file
        :param pimage:
        :return:
        """
        self.trans = False
        self.normalized = False
        try:
            self.data = Image.open(pimage)
            self.fname = pimage
            self.corrupted = False
        except IOError:
            # Empty file
            self.corrupted = True

    def is_correct(self):
        """
        Returns of the image is correct or has enough quality

        Different checks can be done to the image, for now only the "service not available" is check in the init method

        :return:
        """
        # Do image checking

        if self.data is not None and not self.trans and not self.normalized:
            self.correct = True
            # Checks if it is no service image for BCN
            self.correct = self.correct and not np.all(np.asarray(self.data) == self.bcnnoserv)
            # Apply a transformation to the image to check if the file is corrupted
            try:
               img = self.data.crop((5, 5, self.data.size[0] - 5, self.data.size[1] - 5))
               img = self.data.resize((int(0.5 * self.data.size[0]), int(0.5 * self.data.size[1])), PIL.Image.ANTIALIAS)
            except IOError:
                print(self.fname)
                self.correct = False

        else:
            raise Exception('Image already transformed')
        return self.correct

    def transform_image(self, z_factor=None, crop=(0, 0, 0, 0)):
        """
        Performs the transformation of the image

        The idea is to check if the image is correct before applying the transformation

        :param z_factor:
        :param crop:
        :return:
        """
        try:
            if self.data is not None and not self.trans and not self.normalized:
                img = self.data.crop((crop[0], crop[2], self.data.size[0] - crop[1], self.data.size[1] - crop[3]))
                if z_factor is not None:
                    img = img.resize((int(z_factor * img.size[0]), int(z_factor * img.size[1])), PIL.Image.ANTIALIAS)
                self.data = np.asarray(img.convert('RGB'))
                self.trans = True
            else:
                raise Exception('Image not loaded or already transformed')
            return self.data
        except IOError:
            print(self.fname)

    def normalize(self):
        """
        Normalizes the values of the image to the interval [0-1]
        :return:
        """
        if self.data is not None and self.trans and not self.normalized:
            img = self.data / 255.0
            self.data = img
        else:
            raise Exception('Image not yet transformed')

    def get_data(self):
        """
        Returns the data from the image, if it is transformed already
        :return:
        """
        if self.data is not None and self.trans:
            return self.data
        else:
            raise Exception('Image not yet transformed')

    def data_augmentation(self):
        """
        Generates variations of the original image, now does nothing

        Possibilities: horizontal flip, (zoom in + crop) parts of the image
        :return:
        """
        if self.data is not None and self.trans:
            flipped = np.fliplr(self.data)
        else:
            raise Exception('Image not yet transformed')

        return [flipped]

    def flip_image(self):
        """
        Return a lR flip of the image

        Possibilities: horizontal flip, (zoom in + crop) parts of the image
        :return:
        """
        if self.data is not None and self.trans:
            return np.fliplr(self.data)
        else:
            raise Exception('Image not yet transformed')

    def show(self):
        """
        Plots the data from the image
        :return:
        """
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        sp1 = fig.add_subplot(1, 1, 1)
        sp1.imshow(self.data)
        plt.show()
        plt.close()

    def greyscale_histo(self, nbins, percentage):
        """
         Given the data of an image, combines the RBG components transforming it to greyscale using rec 601 luma
         (R*0.299+G*0.587+B*0.114) divides the values in in nbins equal sized bins, counts its frequencies  returns if
          the bin with higher count represents more than a percentage of the image
        :param perc:
        :return:
        """
        if self.data is not None and self.trans:
            cutout = int(self.data.shape[0] * self.data.shape[1] * (percentage/100.0))
            mprod = 0.299 * self.data[:, :, 0] + 0.587 * self.data[:, :, 1] + 0.114 * self.data[:, :, 0]
            hist, bins = np.histogram(mprod.ravel(), bins=nbins)
            return np.max(hist) > cutout
        else:
            raise Exception('Image not yet transformed')

    # def entropy_bad_pixels(self, nbins, cutoff):
    #     """
    #     Given the data of an image, binarizes the RGB components in nbins equal sized bins, counts its frequencies
    #     (in a 3D Matrix) computes the histogram entropy and returns if it is more than a threshold
    #     :param perc:
    #     :return:
    #     """
    #     if self.data is not None and self.trans:
    #         npixels = self.data.shape[0] * self.data.shape[1]
    #         pixels_dict = {}
    #
    #         for i in range(self.data.shape[0]):
    #             for j in range(self.data.shape[1]):
    #                 coord = (int(self.data[i,j,0] * nbins),int(self.data[i,j,1] * nbins),int(self.data[i,j,2] * nbins))
    #                 if coord in pixels_dict:
    #                     pixels_dict[coord] += 1
    #                 else:
    #                     pixels_dict[coord] = 1
    #         ent = 0.0
    #         for c in pixels_dict:
    #             ent -= float(pixels_dict[c])/npixels * np.log2(float(pixels_dict[c])/npixels)
    #         return ent < cutoff
    #     else:
    #         raise Exception('Image not yet transformed')

    def entropy(self, nbins):
        """
        Given the data of an image, binarizes the RGB components in nbins equal sized 256/bins, counts its frequencies
        (in a 3D Matrix) computes the histogram entropy and returns if it is more than a threshold
        nbins must be a power of 2
        :param perc:
        :return:
        """
        if np.log2(nbins)!=int(np.log2(nbins)):
            raise Exception('Nbins must be a power of 2')
        if self.data is not None and self.trans:
            npixels = self.data.shape[0] * self.data.shape[1]
            data = self.data/nbins
            imgR = data[:,:,0].ravel()
            imgG = data[:,:,1].ravel()
            imgB = data[:,:,2].ravel()
            counts = np.zeros((256/nbins, 256/nbins, 256/nbins), dtype=float)+0.00000000001 # avoid log(0)

            for i in range(imgR.shape[0]):
                counts[imgR[i], imgG[i], imgB[i]] += 1
            counts /= npixels
            lcounts = np.log2(counts)
            ent = - lcounts * counts
            return np.sum(ent)
        else:
            raise Exception('Image not yet transformed')

    def var_laplacian(self):
        """
        Returns the variance of the laplacian filter applied to the image
        :return:
        """
        if self.data is not None and self.trans:
            gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        else:
            raise Exception('Image not yet transformed')

    # def var_laplacian_np(self):
    #     """
    #     Returns the variance of the laplacian filter applied to the image
    #     :return:
    #     """
    #     if self.data is not None and self.trans:
    #         gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
    #         return cv2.Laplacian(gray, cv2.CV_64F).var()
    #     else:
    #         raise Exception('Image not yet transformed')

    def histogram(self):
        """
        plots the colors histograms of the rgb channels of the image

        :return:
        """
        if self.data is not None and self.trans:
            fig = plt.figure()
            fig.set_figwidth(300)
            fig.set_figheight(100)
            sp1 = fig.add_subplot(1, 2, 1)
            sp1.imshow(self.data)
            # rec 601 luma
            mprod = (self.data[:, :, 0] * 0.299) + (0.587 * self.data[:, :, 1]) + (0.114 * self.data[:, :, 0])
            hist, bins = np.histogram(mprod.ravel(), bins=50)
            sp2 = fig.add_subplot(1, 2, 2)
            sp2.plot(bins[:-1], hist, 'r')
            plt.show()
            plt.close()
        else:
            raise Exception('Image not yet transformed')

if __name__ == '__main__':
    # from Traffic.Config.Constants import cameras_path
    # limg = sorted(glob.glob(cameras_path + '/20161202/*.gif'))
    # image = TrImage()
    # for img in limg:
    #     image.load_image(img)
    #     image.transform_image(crop=(5, 5, 5, 5))
    #     if image.entropy(8) <4:
    #         image.histogram()
    #         print 'D'

    # limg = sorted(glob.glob(cameras_path + '20161203/*.gif'))
    # image = TrImage()
    # entropies = []
    # for img in limg:
    #     print img
    #     image.load_image(img)
    #     if image.is_correct():
    #         entropies.append(image.var_laplacian(crop=(5, 5, 5, 5)))
    # fig = plt.figure()
    # fig.set_figwidth(300)
    # fig.set_figheight(100)
    # sp2 = fig.add_subplot(1, 1, 1)
    # hist, bins = np.histogram(entropies, bins=50)
    # sp2.plot(bins[:-1], hist, 'r')
    # plt.show()
    # plt.close()

    # limg = sorted(glob.glob(cameras_path + '/20170112/*.gif'))
    # image = TrImage()
    # for img in limg:
    #     badlap = badent = badp = False
    #     image.load_image(img)
    #     image.transform_image(crop=(5, 5, 5, 5))
    #     lap = image.var_laplacian()
    #     badp = image.greyscale_histo(150, 30)
    #     if  lap < 1000:
    #         badlap = True
    #     elif lap < 1300:
    #         ent = image.entropy(20)
    #         badent = ent < 4
    #
    #     if badp or badlap or badent:
    #         print badp, badlap, badent, img.split('/')[-1]
    #         if badp and (not badlap and not badent):
    #             image.histogram()
    #         #


    # image.load_image(cameras_path + '/20161101/201611011453-RondaLitoralZonaFranca.gif')
    # image.load_image(cameras_path + '/20161101/201611010004-PlPauVila.gif')
    # image.show()
    # image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
    # image.histogram()
    # if image.is_correct():
    #     im = image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
    #     print im.shape
    #     image.show()
    #
    #     nimg = TrImage()
    #     nimg.data = image.data_augmentation()[0]
    #     nimg.show()

    image = TrImage()
    image.load_image('/home/bejar/storage/Data/Traffic/Cameras/20170201/201702011001-RondaLitoralBonPastor.gif')
    print(image.is_correct())
    image.show()
    if image.is_correct():
        im = image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
        print(im.shape)
        image.show()
    else:
        print('Incorrect Image')

    # import time
    #
    # image = TrImage()
    # image.load_image('/home/bejar/storage/Data/Traffic/Cameras/20170201/201702011001-RondaLitoralBonPastor.gif')
    # print image.corrupted
    # image.transform_image(crop=(5, 5, 5, 5))
    # itime = time.time()
    # print image.entropy(16)
    # ftime = time.time()
    # print(ftime - itime)
    # itime = time.time()
    #
    # print image.fast_entropy(16)
    # ftime = time.time()
    # print(ftime - itime)
    # print image.trans
