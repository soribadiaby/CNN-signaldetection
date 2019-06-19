# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:59:20 2019

@author: Soriba
"""
import os
import numpy as np

#max_y est le nombre de pixels
path_5000='5000ms'

classes = ['AIS','DMR','EDACS48','EDACS96','NXDN48','NXDN96','ProtocolA','ProtocolB',
 'ProtocolC','ProtocolD']  #np.load('classes.npy')

burst_durations = [26.6,1800,60,30,168,84,100,100,'continu','continu']

bandwidths = [14.4,12.5,6.25,12.5,6.25,12.5,16.4,4.6,16.4,11.6]

def get_px(max_x,max_y):
    
    bandwidths_px=[]
    burst_durations_px=[]
    for burst, bandwidth in zip(burst_durations, bandwidths):
        if burst=='continu':
            burst_durations_px.append(max_y)
        else:
            burst_durations_px.append(np.ceil(max_y*burst/5000))
        
        bandwidths_px.append(np.ceil(max_x*bandwidth/1e3))  #On a un span de 1e6 et une bande passante en kHz, donc en divisent il reste 1e3
    return bandwidths_px, burst_durations_px
        

#caracterisiques = dict(zip(classes, zip(burst_durations_px,bandwidths_px))) #dictionnaire contenant commes cles les classes, et les valeurs sont des tuples (burst, band)

def get_burst_start(classe, filename, max_y, y1): #les debuts des bursts sont dans des fichiers .txt dans le dossier 5000ms
    file = os.path.join(path_5000, classe, filename)
    try:
        burst_starts=open(file, 'r').read().splitlines()
        
    except IOError: #pour les signaux 'continus'
        return [y1]
    
    burst_starts_px=[]
    for k in range(len(burst_starts)):
        start_time = int(burst_starts[k])
        start_time_px = y1 + np.floor(max_y*start_time/5000) #le debut n'est pas à zero à cause de la bande blanche
        burst_starts_px.append(start_time_px)
        
    return burst_starts_px
    
    
    


