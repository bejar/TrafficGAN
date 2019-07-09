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

class Config:
    datapath = None

    def __init__(self):
        """
        Read the configuration file that is in the current directory
        """
        config = json.loads('traffic.json')

        self.datapath = config['datapath']

    def info(self):
        """
        print the information in the configuration object
        :return:
        """

        print(f'DATAPATH={self.datapath}')
