"""
.. module:: Cameras

Cameras
*************

:Description: Cameras

 List of camera names from Barcelona

:Authors: bejar
    

:Version: 

:Created on: 27/10/2016 8:34 

"""

__author__ = 'bejar'
# All the public traffic cameras of Barcelona city
Cameras = [
    'AragoPgGracia',
    'BalmesGranVia',
    'BalmesMitre',
    'DiagonalCiutatdeGranada',
    'DiagonalMCristina',
    'GranViaBacRoda',
    'GranViaMarina',
    'MarinaPujades',
    'MeridianaFelipII',
    'MeridianaRioJaneiro',
    'PlAntonioLopez',
    'PlCatalunya',
    'PlCerda',
    'PlEspanya',
    'PlMolina',
    'PlPaissosCatalans',
    #    'PlPauVila',
    'PlUrquinaona',
    'RondaLitoralBadajoz',
    'RondadeDaltCrtaEsplugues',
    'RondadeDaltMeridiana',
    'RondadeDaltSantGervasi',
    'RondadeDaltVelodrom',
    'RondaLitoralBonPastor',
    'RondaLitoralMollFusta',
    'RondaLitoralZonaFranca',
    'TunelRovira'
]

# list of the "good" cameras (the commented ones are out of service or pointing mostly to trees
Cameras_ok = [
    'AragoPgGracia',
    'BalmesGranVia',
    'BalmesMitre',
    #    'DiagonalCiutatdeGranada',
    #    'DiagonalMCristina',
    'GranViaBacRoda',
    'GranViaMarina',
    'MarinaPujades',
    'MeridianaFelipII',
    'MeridianaRioJaneiro',
    'PlAntonioLopez',
    'PlCatalunya',
    'PlCerda',
    'PlEspanya',
    'PlMolina',
    'PlPaissosCatalans',
    #    'PlPauVila',
    'PlUrquinaona',
    'RondaLitoralBadajoz',
    'RondadeDaltCrtaEsplugues',
    'RondadeDaltMeridiana',
    'RondadeDaltSantGervasi',
    'RondadeDaltVelodrom',
    'RondaLitoralBonPastor',
    #    'RondaLitoralMollFusta',
    'RondaLitoralZonaFranca',
    'TunelRovira'
]

# Longitude, latitude of the cameras
CamCoord = {
    'AragoPgGracia': (2.16487462, 41.39225298),
    'BalmesGranVia': (2.16560024, 41.3871296),
    'BalmesMitre': (2.14077336, 41.40444905),
    'DiagonalCiutatdeGranada': (2.19057765, 41.40415691),
    'DiagonalMCristina': (2.12703713, 41.38836632),
    'GranViaBacRoda': (2.1959603, 41.411915),
    'GranViaMarina': (2.181249530000059, 41.39914869),
    'MarinaPujades': (2.189120389999971, 41.39305967),
    'MeridianaFelipII': (2.1871113, 41.42029817),
    'MeridianaRioJaneiro': (2.18641026, 41.44309219),
    'PlAntonioLopez': (2.18207217, 41.38141543),
    'PlCatalunya': (2.16863241, 41.38749875),
    'PlCerda': (2.13587689, 41.36499236),
    'PlEspanya': (2.14976484, 41.37516851),
    'PlMolina': (2.14766551, 41.40120644),
    'PlPaissosCatalans': (2.14417036, 41.3810396),
    #    'PlPauVila': (2.18532062,41.38165624),
    'PlUrquinaona': (2.17228212, 41.38890733),
    'RondaLitoralBadajoz': (2.20302198, 41.39294027),
    'RondadeDaltCrtaEsplugues': (2.10713156, 41.38865464),
    'RondadeDaltMeridiana': (2.1877678, 41.44698951),
    'RondadeDaltSantGervasi': (2.13618326, 41.41525263),
    'RondadeDaltVelodrom': (2.15162852, 41.43769067),
    'RondaLitoralBonPastor': (2.20924092, 41.43597442),
    'RondaLitoralMollFusta': (2.17973115, 41.37724461),
    'RondaLitoralZonaFranca': (2.1468751, 41.34935557),
    'TunelRovira': (2.1581906500000514, 41.42382369)
}
