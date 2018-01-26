#!/usr/bin/env python3

""" 
Efficient, local elevation lookup using intermediate tile representation 
of world-wide SRTM elevation data.

Example:
import elevation_grid as eg
el = eg.get_elevations(np.array([[49.3,123.1]]))

import matplotlib.pyplot as plt
lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]
plt.pcolormesh(lons,lats,elevs,cmap='terrain')
plt.colorbar()
plt.show()
"""

import numpy as np
import os

scriptpath = os.path.dirname(os.path.realpath(__file__))
elev_fname = os.path.join(scriptpath, 'elevations_latlon.npy')
tiles_fname = os.path.join(scriptpath, 'tiles_latlon.npy')

lat_ranges = np.arange(-90,90,10)
lon_ranges = np.arange(-180,180,10)
elevgrid = None

def make_elevation_grid():
    """ Uses SRTM.py to create an intermediate elevation tile representation and
    concatenates the tiles into a single array that can be indexed via latitude and longitude.
    Note, this takes a long time. Don't run this if the elevation grid is already available.
    """
    def cleanup_elevation_grid():
        ta = [np.concatenate(tr,axis=1) for tr in tiles_latlon]
        ta = np.concatenate(ta)
        ta[np.isnan(ta)] = 0
    np.save(elev_fname)

    import srtm
    try:
        tiles_latlon = np.load('tiles_latlon.npy')
    except:
        print('Creating list of empty tiles')
        tiles_latlon = [[None for _ in range(len(lon_ranges))] for _ in range(len(lat_ranges))]
    for k, lati in enumerate(lat_ranges):
        ed = srtm.get_data()
        for l, loti in enumerate(lon_ranges):
            print(lati, loti)
            if tiles_latlon[k][l] is None:
                try:
                    tiles_latlon[k][l] = ed.get_image((100,100),
                                                      (lati,lati+10),
                                                      (loti,loti+10),
                                                      10000,
                                                      mode='array')
                except:
                    print('Error producing tile {}, {}'.format(lati,loti))
                    pass
                np.save('tiles_latlon.npy', tiles_latlon)
    cleanup_elevation_grid()
    # broken_tiles = ['N21E035.hgt', 'N22E035.hgt', 'N24E035.hgt', 'N25E035.hgt', 'N26E035.hgt', 
    #                 'N27E035.hgt', 'N28E035.hgt', 'N27E039.hgt', 'N28E035.hgt', 'N28E039.hgt',
    #                 'N29E039.hgt', ]

import gzip
try:
    fh = gzip.open('elevations_latlon.npy.gz','rb')
except:
    fh = open('elevations_latlon.npy','rb')
elevgrid = np.load(fh)
fh.close()

def get_elevations(latlon):
    """For latlon being a N x 2 np.array of latitude, longitude pairs, output an
    array of length N giving the elevations in meters
    """
    lli = ((latlon + (90,180))*10).astype(int)
    return elevgrid[lli[:,0],lli[:,1]]

