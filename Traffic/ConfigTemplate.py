"""
.. module:: Config

Config
*************

:Description: Config

    

:Authors: bejar
    

:Version: 

:Created on: 09/07/2019 13:11 

"""

import json

__author__ = 'bejar'

configpath = '/home/bejar/PycharmProjects/TrafficGAN/'

class Config:
    datapath = None
    infopath = None
    cameraspath = None
    output_dir = None

    def __init__(self):
        """
        Read the configuration file that is in the current directory
        """

        fp = open(f'{configpath}/traffic.json', 'r')
        s = ''

        for l in fp:
            s += l
        config = json.loads(s)

        self.datapath = config['datapath']
        self.infopath = config['infopath']
        self.cameraspath = config['cameraspath']
        self.output_dir = config['outputdir']

    def info(self):
        """
        print the information in the configuration object
        :return:
        """

        print(f'DATAPATH={self.datapath}')
        print(f'INFOPATH={self.infopath}')
        print(f'CAMERASPATH={self.cameraspath}')
        print(f'OUTPUTDIR={self.output_dir}')
