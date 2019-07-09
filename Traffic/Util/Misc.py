"""
.. module:: Misc

Misc
*************

:Description: Misc

    Miscelaneous functions

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 9:41 

"""

import json

__author__ = 'bejar'


def recoding_dictionary(recode):
    """
    Transforms a recoding string into a recoding dictionary
    :param recode:
    :return:
    """
    code = recode.split(',')
    rec = {}
    for c in code:
        k, v = c.split('|')
        rec[int(k)] = int(v)
    return rec


def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def transweights(weights):
    """
    Transforms class weights format from json to python
    :param weights:
    :return:
    """
    wtrans = {}
    for v in weights:
        wtrans[str(v)] = weights[v]
    return wtrans


def detransweights(weights):
    """
    Transforms class weights format from python to json
    :param weights:
    :return:
    """
    wtrans = {}
    for v in weights:
        wtrans[int(v)] = weights[v]
    return wtrans


def list_days_generator(year, month, iday, fday):
    """
    Generates a list of days

    :param year:
    :param month:
    :param iday:
    :param fday:
    :return:
    """
    ldays = []
    for v in range(iday, fday + 1):
        ldays.append("%04d%02d%02d" % (year, month, v))
    return ldays


def list_range_days_generator(idate, fdate):
    """
    Generates a list of days between two dates

    Note: Not all cases have been contemplated, STILL TO BE COMPLETED

    :param idate:
    :param fdate:
    :return:
    """

    ndays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # no leap year
    irange = (int(idate[0:4]), int(idate[4:6]), int(idate[6:8]))
    erange = (int(fdate[0:4]), int(fdate[4:6]), int(fdate[6:8]))

    if 0 < irange[1] <= 12 and 0 < erange[1] <= 12:
        if irange[0] == erange[0]:  # Same Year
            if irange[1] == erange[1]:  # Same Month
                ldays = list_days_generator(irange[0], irange[1], irange[2], erange[2])
            else:  # Several months
                ldays = list_days_generator(irange[0], irange[1], irange[2], ndays[irange[1]])
                for i in range(irange[1] + 1, erange[1]):
                    ldays.extend(list_days_generator(irange[0], i, 1, ndays[i]))
                ldays.extend(list_days_generator(irange[0], erange[1], 1, erange[2]))
        else:
            if erange[0] - irange[0] <= 1:
                ldays = list_days_generator(irange[0], irange[1], irange[2], ndays[irange[1]])
                for i in range(irange[1] + 1, 13):
                    ldays.extend(list_days_generator(irange[0], i, 1, ndays[i]))
                for i in range(1, erange[1]):
                    ldays.extend(list_days_generator(erange[0], i, 1, ndays[i]))
                ldays.extend(list_days_generator(erange[0], erange[1], 1, erange[2]))


            else:
                raise Exception('More than a year')

    else:
        raise Exception('Wrong month number')

    return ldays


def name_days_file(ldays):
    """
    Generates a string with the dates

    :param ldays:
    :return:
    """
    return ldays[0] + '-' + ldays[-1]


def dist_time(time1, time2):
    """
    distance between two hours in minutes

    :param time1:
    :param time2:
    :return:
    """
    t1 = (time1 % 100) + (60 * ((time1 // 100) % 100))
    t2 = (time2 % 100) + (60 * ((time2 // 100) % 100))
    return t2 - t1

def get_hour(time1):
    """
    returns the hour of an integer coded yyyymmddhhmm

    :param time1:
    :param time2:
    :return:
    """
    return (time1 // 100) % 100


if __name__ == '__main__':
    # print (list_range_days_generator('20161101', '20170205'))
    print(recoding_dictionary("0|0,1|1,2|2,3|3,4|4"))
