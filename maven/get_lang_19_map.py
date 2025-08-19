# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:57:50 2024

@author: Win
"""
from scipy.io import readsav
import numpy as np
import os
current_dir = os.path.dirname(__file__)
sav_fname = os.path.join(current_dir, "Langlais2018_400km_0.5deg.sav")

def get_lang_19_map():
    langlais_br = readsav(sav_fname)['langlais']['br'][0]
    langlais_bt = readsav(sav_fname)['langlais']['bt'][0]
    langlais_bp = readsav(sav_fname)['langlais']['bp'][0]
    langlais_elon = readsav(sav_fname)['langlais']['elon'][0]
    langlais_lat = readsav(sav_fname)['langlais']['lat'][0]
    langlais_btot=np.sqrt(langlais_br**2+langlais_bt**2+langlais_bp**2);
    lon2d_langlais, lat2d_langlais = np.meshgrid(langlais_elon, langlais_lat)
    res={'lon':lon2d_langlais,
         'lat':lat2d_langlais,
         'br':langlais_br,
         'bt':langlais_bt,
         'bp':langlais_bp,
         'btot':langlais_btot}
    return res

