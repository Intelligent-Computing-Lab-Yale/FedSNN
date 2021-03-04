#!/usr/bin/env python

'''
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.
'''


from __future__ import print_function
import os, sys, time, argparse
import multiprocessing as mp
from multiprocessing import Queue
from queue import Empty
import numpy as np
import h5py
from copy import deepcopy
from view import HDF5Stream, MergedStream
from datasets import HDF5
from interfaces.caer import DVS_SHAPE, unpack_data

print("Found cpu cores:", mp.cpu_count())

export_data_vi = {
        'steering_wheel_angle',
        'brake_pedal_status',
        'accelerator_pedal_position',
        'engine_speed',
        'vehicle_speed',
        'windshield_wiper_status',
        'headlamp_status',
        'transmission_gear_position',
        'torque_at_transmission',
        'fuel_level',
        'high_beam_status',
        'ignition_status',
        #'lateral_acceleration',
        'latitude',
        'longitude',
        #'longitudinal_acceleration',
        'odometer',
        'parking_brake_status',
        #'fine_odometer_since_restart',
        'fuel_consumed_since_restart',
    }

export_data_dvs = {
        'dvs_accum',
        'dvs_split',
        'dvs_channels',
        'aps_frame',
    }

export_data = export_data_vi.union(export_data_dvs)


def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8

def get_progress_bar():
    try:
        from tqdm import tqdm
    except ImportError:
        print("\n\nNOTE: For an enhanced progress bar, try 'pip install tqdm'\n\n")
        class pbar():
            position=0
            def close(self): pass
            def update(self, increment):
                self.position += increment
                print('\r{}s done...'.format(self.position)),
        def tqdm(*args, **kwargs):
            return pbar()
    return tqdm(total=(tstop-tstart)/1e6, unit_scale=True)

def raster_evts(data, seperate_dvs_channels = False, split_timesteps = False, timesteps = 10):
    _histrange = [(0, v) for v in DVS_SHAPE]
    if split_timesteps:
        img = []
        data_chunks = np.array_split(data, timesteps)
        for i in range(timesteps):
            pol_on = data_chunks[i][:,3] == 1
            pol_off = np.logical_not(pol_on)
            img_on, _, _ = np.histogram2d(
                    data_chunks[i][pol_on, 2], data_chunks[i][pol_on, 1],
                    bins=DVS_SHAPE, range=_histrange)
            img_off, _, _ = np.histogram2d(
                    data_chunks[i][pol_off, 2], data_chunks[i][pol_off, 1],
                    bins=DVS_SHAPE, range=_histrange)
            img.append(np.stack([img_on.astype(np.int16), img_off.astype(np.int16)]))
        img = np.array(img)
        # print(img.shape)
        return img

    pol_on = data[:,3] == 1
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            data[pol_on, 2], data[pol_on, 1],
            bins=DVS_SHAPE, range=_histrange)
    img_off, _, _ = np.histogram2d(
            data[pol_off, 2], data[pol_off, 1],
            bins=DVS_SHAPE, range=_histrange)
    if seperate_dvs_channels:
        return np.stack([img_on.astype(np.int16), img_off.astype(np.int16)])
    return (img_on - img_off).astype(np.int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--tstart', type=int, default=0)
    parser.add_argument('--tstop', type=int)
    parser.add_argument('--binsize', type=float, default=0.1)
    parser.add_argument('--update_prog_every', type=float, default=0.01)
    parser.add_argument('--export_aps', type=int, default=1)
    parser.add_argument('--export_dvs', type=int, default=1)
    parser.add_argument('--out_file', default='')
    parser.add_argument('--seperate_dvs_channels', action='store_true')
    parser.add_argument('--split_timesteps', action='store_true')
    parser.add_argument('--timesteps', type=int, default=10)
    args = parser.parse_args()

    f_in = HDF5Stream(args.filename, export_data_vi.union({'dvs'}))
    m = MergedStream(f_in)

    fixed_dt = args.binsize > 0
    tstart = int(m.tmin + 1e6 * args.tstart)
    tstop = (m.tmin + 1e6 * args.tstop) if args.tstop is not None else m.tmax
    print('start/stop timestamp', tstart, tstop)
    print('recording duration', (m.tmax - m.tmin) * 1e-6, 's')

    # find start position
    m.search(tstart)

    #create output file
    dtypes = {k: float for k in export_data.union({'timestamp'})}
    if args.export_aps:
        dtypes['aps_frame'] = (np.uint8, DVS_SHAPE)
    if args.export_dvs:
        dtypes['dvs_split'] = (np.int16, (args.timesteps, 2, DVS_SHAPE[0], DVS_SHAPE[1]))
        dtypes['dvs_channels'] = (np.int16, (2, DVS_SHAPE[0], DVS_SHAPE[1]))
        dtypes['dvs_accum'] = (np.int16, DVS_SHAPE)

    outfile = args.out_file or args.filename[:-5] + '_export.hdf5'
    f_out = HDF5(outfile, dtypes, mode='w', chunksize=8, compression='gzip')

    current_row = {k: 0 for k in dtypes}
    if args.export_aps:
        current_row['aps_frame'] = np.zeros(DVS_SHAPE, dtype=np.uint8)
    if args.export_dvs:
        current_row['dvs_split'] = np.zeros((args.timesteps, 2, DVS_SHAPE[0], DVS_SHAPE[1]), dtype=np.int16)
        current_row['dvs_channels'] = np.zeros((2, DVS_SHAPE[0], DVS_SHAPE[1]), dtype=np.int16)
        current_row['dvs_accum'] = np.zeros(DVS_SHAPE, dtype=np.int16)

    pbar = get_progress_bar()
    sys_ts, t_pre, t_offset, ev_count, pbar_next = 0, 0, 0, 0, 0
    while m.has_data and sys_ts <= tstop*1e-6:
        try:
            sys_ts, d = m.get()
        except Empty:
            # wait for queue to fill up
            time.sleep(0.01)
            continue
        if not d:
            # skip unused data
            continue
        if d['etype'] == 'special_event':
            unpack_data(d)
            if any(d['data'] == 0): # this is a timestamp reset
                print('ts reset detected, setting offset', current_row['timestamp'])
                t_offset += current_row['timestamp']
                #NOTE the timestamp of this special event is not meaningful
                continue
        if d['etype'] in export_data_vi:
            current_row[d['etype']] = d['data']
            continue
        if t_pre == 0 and d['etype'] in ['frame_event', 'polarity_event']:
            print('resetting t_pre (first %s)' % d['etype'])
            t_pre = d['timestamp'] + t_offset
        if d['etype'] == 'frame_event' and args.export_aps:
            if fixed_dt:
                while t_pre + args.binsize < d['timestamp'] + t_offset:
                    # aps frame is not in current bin -> save and proceed
                    f_out.save(deepcopy(current_row))
                    current_row['dvs_accum'] = 0
                    current_row['dvs_channels'] = 0
                    current_row['dvs_split'] = 0
                    current_row['timestamp'] = t_pre
                    t_pre += args.binsize
            else:
                current_row['timestamp'] = d['timestamp'] + t_offset
            current_row['aps_frame'] = filter_frame(unpack_data(d))
            #current_row['timestamp'] = t_pre
            #JB: I don't see why the previous line should make sense
            continue
        if d['etype'] == 'polarity_event' and args.export_dvs:
            unpack_data(d)
            times = d['data'][:, 0] * 1e-6 + t_offset
            num_evts = d['data'].shape[0]
            offset = 0
            if fixed_dt:
                # fixed time interval bin mode
                num_samples = int(np.ceil((times[-1] - t_pre) / args.binsize))
                for _ in range(0, num_samples):
                    # take n events
                    n = (times[offset:] < t_pre + args.binsize).sum()
                    sel = slice(offset, offset + n)
                    x = raster_evts(d['data'][sel], seperate_dvs_channels = args.seperate_dvs_channels, split_timesteps = args.split_timesteps, timesteps = args.timesteps)
                    if args.split_timesteps:
                        current_row['dvs_split'] += x
                    elif args.seperate_dvs_channels:
                        current_row['dvs_channels'] += x
                    else:
                        current_row['dvs_accum'] += x
                    offset += n
                    # save if we're in the middle of a packet, otherwise
                    # wait for more data
                    if sel.stop < num_evts:
                        current_row['timestamp'] = t_pre
                        f_out.save(deepcopy(current_row))
                        current_row['dvs_split'][:,:,:,:] = 0
                        current_row['dvs_channels'][:,:,:] = 0
                        current_row['dvs_accum'][:,:] = 0
                        t_pre += args.binsize
            else:
                # ------------------ Depricated ---------------------#
                # fixed event count mode
                num_samples = np.ceil(-float(num_evts + ev_count)/args.binsize)
                for _ in range(int(num_samples)):
                    n = min(int(-args.binsize - ev_count), num_evts - offset)
                    sel = slice(offset, offset + n)
                    current_row['dvs_frame'] += raster_evts(d['data'][sel], seperate_dvs_channels = args.seperate_dvs_channels)
                    if sel.stop > sel.start:
                        current_row['timestamp'] = times[sel].mean()
                    offset += n
                    ev_count += n
                    if ev_count == -args.binsize:
                        f_out.save(deepcopy(current_row))
                        current_row['dvs_frame'][:,:] = 0
                        ev_count = 0
        pbar_curr = int((sys_ts - tstart * 1e-6) / args.update_prog_every)
        if pbar_curr > pbar_next:
            pbar.update(args.update_prog_every)
            pbar_next = pbar_curr
    pbar.close()
    print('[DEBUG] sys_ts/tstop', sys_ts, tstop*1e-6)
    m.exit.set()
    f_out.exit.set()
    f_out.join()
    print('[DEBUG] output done')
    while not m.done.is_set():
        print('[DEBUG] waiting for merger')
        time.sleep(1)
    print('[DEBUG] merger done')
    f_in.join()
    print('[DEBUG] stream joined')
    m.join()
    print('[DEBUG] merger joined')
    filesize = os.path.getsize(outfile)
    print('Finished.  Wrote {:.1f}MiB to {}.'.format(filesize/1024**2, outfile))

    time.sleep(1)
    os._exit(0)
