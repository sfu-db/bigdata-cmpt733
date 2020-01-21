#!/usr/bin/env python3

""" 
Efficient, local elevation lookup using intermediate tile representation 
of world-wide SRTM elevation data.

Examples:

import elevation_grid as eg
import numpy as np

el = eg.get_elevation(50, -123.1)
print("A place near Whistler, BC is {} m above sea level".format(el))

import matplotlib.pyplot as plt

lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]

plt.pcolormesh(lons,lats,elevs,cmap='terrain')
plt.colorbar()
# plt.show()
"""

import numpy as np
import os

scriptpath = os.path.dirname(os.path.realpath(__file__))
elev_fname = os.path.join(scriptpath, 'elevations_latlon.npy')
tiles_fname = os.path.join(scriptpath, 'tiles_latlon.npy')

tile_size = 100
tile_degrees = 10
lat_ranges = np.arange(-90,90,tile_degrees)
lon_ranges = np.arange(-180,180,tile_degrees)
elevgrid = None

def make_elevation_grid():
    """ Uses SRTM.py to create an intermediate elevation tile representation and
    concatenates the tiles into a single array that can be indexed via latitude and longitude.
    
    Note, this takes a long time and downloads about 62 GB of data.
    Don't run this if the elevation grid is already available.
    """
    def cleanup_elevation_grid():
        """Concatenate tiles_latlon into a single array and replace NaNs with 0"""
        ta = [np.concatenate(tr,axis=1) for tr in tiles_latlon]
        ta = np.concatenate(ta)
        ta[np.isnan(ta)] = 0
        print('Saving elevation array to {}'.format(elev_fname))
        np.save(elev_fname, ta)

    try:
        import srtm
    except:
        print('Install SRTM.py via\n'
              'pip3 install git+https://github.com/tkrajina/srtm.py.git')
        raise
    try:
        print('Resuming construction of tiles from {}'.format(tiles_fname))
        tiles_latlon = np.load(tiles_fname)
    except:
        print('Creating list of empty tiles')
        tiles_latlon = [[None for _ in range(len(lon_ranges))] for _ in range(len(lat_ranges))]
    for k, lati in enumerate(lat_ranges):
        ed = srtm.get_data()
        for l, loti in enumerate(lon_ranges):
            print(lati, loti)
            if tiles_latlon[k][l] is None:      # only compute what we don't yet have
                try:
                    tiles_latlon[k][l] = ed.get_image((tile_size,tile_size),
                                                      (lati,lati+tile_degrees),
                                                      (loti,loti+tile_degrees),
                                                      10000,
                                                      mode='array')
                except:
                    print('Error producing tile {}, {}'.format(lati,loti))
                    pass
                np.save(tiles_fname, tiles_latlon)
    cleanup_elevation_grid()

    # The overall SRTM tile data in ~/.cache/srtm is about 52 GB. It was impossible to download these few:
    # broken_tiles = ['N21E035.hgt', 'N22E035.hgt', 'N24E035.hgt', 'N25E035.hgt', 'N26E035.hgt', 
    #                 'N27E035.hgt', 'N28E035.hgt', 'N27E039.hgt', 'N28E035.hgt', 'N28E039.hgt',
    #                 'N29E039.hgt', ]

# load the preprocess elevation array (about 50 MB uncompressed)
import gzip
try:
    try:
        fh = gzip.open(elev_fname+'.gz','rb')
    except:
        fh = open(elev_fname,'rb')
    elevgrid = np.load(fh)
    fh.close()
except:
    print("Warning: There was a problem initializing the elevation array from {}[.gz]".format(elev_fname))
    print("         Consider to run make_elevation_grid()")

def get_elevations(latlons):
    """For latlons being a N x 2 np.array of latitude, longitude pairs, output an
       array of length N giving the corresponding elevations in meters.
    """
    lli = ((latlons + (90,180))*(float(tile_size)/tile_degrees)).astype(int)
    return elevgrid[lli[:,0],lli[:,1]]

def get_elevation(lat, lon, get_elevations=get_elevations):
    """Lookup elevation in m"""
    return get_elevations(np.array([[lat,lon]]))[0]

import requests
def request_elevations(latlons):
    """Obtain elevations from open-elevation.com"""
    reqjson = dict(locations=[dict(latitude=float(lat),longitude=float(lon)) for lat,lon in latlons])
    r = requests.post('https://api.open-elevation.com/api/v1/lookup', json=reqjson)
    assert r, "Error making open elevation bulk request"
    return [el['elevation'] for el in r.json()['results']]

#-----------------------------------------------------------------------------
import unittest

# from command line: python -m unittest elevation_grid.py

class TestElevationLookups(unittest.TestCase):
    def test_elevations(self):
        """ Compare SRTM against open-elevation.com info """
        tol_m = 100    
        lats, lons = np.meshgrid(np.arange(48, 53, 1), np.arange(118, 122, 1));
        latlons = np.stack([lats.flatten(), lons.flatten()]).T;
        latlons = np.concatenate([latlons, latlons+.1])
        rev = np.array(request_elevations(latlons))
        gev = get_elevations(latlons)
        np.set_printoptions(suppress=True)
        self.assertTrue(abs((rev-gev)).max() < tol_m, np.stack((latlons[:,0],latlons[:,1],rev,gev,rev-gev)).T)
        print('    lat', '    lon', 'open-elev.', 'srtm-array', 'difference')
        print(np.stack((latlons[:,0],latlons[:,1],rev,gev,rev-gev)).T)
