import os, sys
import numpy as np
from file_methods import *
from offline_reference_surface import Offline_Reference_Surface
from offline_surface_tracker import Offline_Surface_Tracker
import csv
import logging
from shutil import copyfile

def correlate_data(data,timestamps):
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    while True:
        try:
            datum = data[data_index]
            ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break

        if datum['timestamp'] <= ts:
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1
        else:
            frame_idx+=1
    return data_by_frame


class Global_Container(object):
    pass


def main(rec_dir, surf):
    
    #copies the surface file from the gaze_on_surface directory to the rec directory
    if os.path.exists(surf):
        copyfile(surf, path + 'surface_definitions')
    else:
        logging.warning("invalid surface file path")

    pupil_data = load_object(os.path.join(rec_dir, "pupil_data_corrected"))
    pupil_list = pupil_data['pupil_positions']
    gaze_list = pupil_data['gaze_positions']
    timestamps = np.load(os.path.join(rec_dir, "world_timestamps.npy"))
    
    g_pool = Global_Container()
    g_pool.rec_dir = rec_dir
    g_pool.timestamps = timestamps
    g_pool.gaze_positions_by_frame = correlate_data(gaze_list, g_pool.timestamps)


    surface_tracker = Offline_Surface_Tracker(g_pool)
    persistent_cache = Persistent_Dict(os.path.join(rec_dir,'square_marker_cache'))
    cache = persistent_cache.get('marker_cache',None)
    camera = load_object('camera')
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
    camera['camera_matrix'] = np.array(camera['camera_matrix'])
    camera['resolution'] = np.array(camera['resolution'])

    for s in surface_tracker.surfaces:
        surface_name = '_'+s.name.replace('/','')+'_'+s.uid
        with open(os.path.join(rec_dir,'gaze_on_surface'+surface_name+'.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(('world_timestamp','world_frame_idx','gaze_timestamp','x_norm','y_norm','on_srf'))    
            for idx, c in enumerate(cache):
                s.locate(c, camera, 0, 0.0)
                ts = timestamps[idx]
                if s.m_from_screen is not None:
                    for gp in s.gaze_on_srf_by_frame_idx(idx,s.m_from_screen):
                        csv_writer.writerow( (ts,idx,gp['base_data']['timestamp'],gp['norm_pos'][0],gp['norm_pos'][1],gp['on_srf']) )
if __name__ == '__main__':
    rootdir = sys.argv[1]
    surf = sys.argv[2]
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"
        if os.path.exists(path + "pupil_data_corrected"):
            main(path, surf)
