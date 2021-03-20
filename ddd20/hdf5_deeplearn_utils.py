import h5py
import numpy as np
#from scipy.misc import imresize
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage import img_as_bool
from collections import defaultdict
import sklearn
import torch
from PIL import Image
import sys

def get_real_endpoint(h5f):
    if h5f['timestamp'][-1] != 0:
        end_len = len(h5f['timestamp'])
    else:
        # Get the last index where it changes, before it becomes zero
        end_len = np.where(abs(np.diff(h5f['timestamp']))>0)[0][-1]+1
    return end_len

def get_common_endpoint(h5f_aps, h5f_dvs):
    return min(get_real_endpoint(h5f_aps), get_real_endpoint(h5f_dvs))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def yield_chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

def check_and_fix_timestamps(h5f):
    all_timestamps = np.array(h5f['timestamp']).copy()
    if np.all(np.diff(all_timestamps) >= 0):
        return
    else:
        print('Broken timestamps found! Fixing...')
        curr_idx = 0
        while curr_idx < len(all_timestamps):
            loop_point = np.where(np.diff(all_timestamps)<0)[0]
            if len(loop_point)==0:
                orig_timestamps = np.array(h5f['timestamp']).copy()
                if 'orig_timestamp' in h5f:
                    del h5f['orig_timestamp']
                h5f['orig_timestamp'] = orig_timestamps
                if 'timestamp' in h5f:
                    del h5f['timestamp']
                h5f['timestamp'] = all_timestamps
                return
            else:
                curr_idx = loop_point[0]
                all_timestamps[curr_idx+1:] += abs(all_timestamps[curr_idx+1]-all_timestamps[curr_idx])


def get_start_stop_contigs(all_idxs):
    # Find where there are jumps
    jumps = np.where(np.diff(all_idxs)>1)[0]
    starts = [0,]+list(jumps+1)
    stops = list(jumps+1)+[len(all_idxs)]
    return starts, stops

def calc_data_mean(h5f, group_key, chunksize=1024*10, axes=(0), force=False):
    ep = get_real_endpoint(h5f)
    # Make chunks of contiguous blocks
    train_idxs = list(h5f['train_idxs'])
    starts, stops = get_start_stop_contigs(h5f['train_idxs'])

    # Use a nested list comprehension over contiguous blocks of chunks of images, summing all.
    #     Chunking on ranges (start:stop) is actually a fair bit faster than chunking on
    #     individual indices.
    new_mean = np.sum([np.sum([group.astype('double').sum(axis=axes)
                       for group in chunker(h5f[group_key][start:stop], chunksize)], axis=0)
                            for start, stop in zip(starts, stops)], axis=0)
    new_mean = new_mean.astype('float64') / len(h5f[group_key][train_idxs])
    # Replace
    if force and group_key+'_mean' in h5f:
        del h5f[group_key+'_mean']
    h5f.create_dataset(group_key+'_mean', data=np.array(new_mean))
    return

def calc_data_std(h5f, group_key, chunksize=1024*10, axes=(0), force=False):
    ep = get_real_endpoint(h5f)
    train_idxs = list(h5f['train_idxs'])
    mean_val = h5f[group_key+'_mean']
    starts, stops = get_start_stop_contigs(h5f['train_idxs'])

    # Do it as parallel as possible, using the mean to upcast it to double
    sum_sq_val = np.sum([np.sum([np.sum((group-mean_val)**2,axis=axes)
                                    for group in chunker(h5f[group_key][start:stop], chunksize)], axis=0)
                                        for start, stop in zip(starts, stops)], axis=0)
    std_val = np.sqrt(sum_sq_val/len(h5f[group_key][train_idxs]))
    std_val[std_val==0] = 1.
    # Replace
    if force and group_key+'_std' in h5f:
        del h5f[group_key+'_std']
    h5f.create_dataset(group_key+'_std', data=np.array(std_val))
    return

def build_simul_train_test_split(h5f_aps, h5f_dvs_accum, h5f_dvs_split, train_div=5*60, test_div=1*60, force=False):
    ep = get_common_endpoint(h5f_aps, h5f_dvs_accum)
    train_idxs, test_idxs = [], []
    curr_idx = 0
    curr_ts_aps = h5f_aps['timestamp'][0]
    curr_ts_dvs_accum = h5f_dvs_accum['timestamp'][0]
    curr_ts_dvs_split = h5f_dvs_split['timestamp'][0]
    curr_ts = max(curr_ts_aps, curr_ts_dvs_accum, curr_ts_dvs_split)

    while curr_idx < len(h5f_aps['timestamp'][:ep]) and curr_idx < len(h5f_dvs_accum['timestamp'][:ep]) and curr_idx < len(h5f_dvs_split['timestamp'][:ep]):
        # Get train indexes
        stop_idx = np.where(h5f_aps['timestamp'][curr_idx:ep] > curr_ts+train_div)[0]
        stop_idx = stop_idx[0] if np.any(stop_idx) else ep-curr_idx
        train_idxs.extend(range(curr_idx, curr_idx+stop_idx))
        curr_ts = curr_ts+train_div
        curr_idx = curr_idx+stop_idx
        # Get test indexes
        stop_idx = np.where(h5f_aps['timestamp'][curr_idx:ep] > curr_ts+test_div)[0]
        stop_idx = stop_idx[0] if np.any(stop_idx) else ep-curr_idx
        test_idxs.extend(range(curr_idx, curr_idx+stop_idx))
        curr_ts = curr_ts+test_div
        curr_idx = curr_idx+stop_idx
    # Replace as necessary
    if force and (('train_idxs' in h5f_aps) or ('train_idxs' in h5f_dvs_accum) or ('train_idxs' in h5f_dvs_split)):
        del h5f_aps['train_idxs']
        del h5f_dvs_accum['train_idxs']
        del h5f_dvs_split['train_idxs']
    h5f_aps.create_dataset('train_idxs', data=np.array(train_idxs))
    h5f_dvs_accum.create_dataset('train_idxs', data=np.array(train_idxs))
    h5f_dvs_split.create_dataset('train_idxs', data=np.array(train_idxs))
    if force and (('test_idxs' in h5f_aps) or ('test_idxs' in h5f_dvs_accum) or ('test_idxs' in h5f_dvs_split)):
        del h5f_aps['test_idxs']
        del h5f_dvs_accum['test_idxs']
        del h5f_dvs_split['test_idxs']
    h5f_aps.create_dataset('test_idxs', data=np.array(test_idxs))
    h5f_dvs_accum.create_dataset('test_idxs', data=np.array(test_idxs))
    h5f_dvs_split.create_dataset('test_idxs', data=np.array(test_idxs))
    return

def build_train_test_split(h5f, train_div=5*60, test_div=1*60, force=False):
    ep = get_real_endpoint(h5f)
    train_idxs, test_idxs = [], []
    curr_idx, curr_ts = 0, h5f['timestamp'][0]
    while curr_idx < len(h5f['timestamp'][:ep]):
        # Get train indexes
        stop_idx = np.where(h5f['timestamp'][curr_idx:ep] > curr_ts+train_div)[0]
        stop_idx = stop_idx[0] if np.any(stop_idx) else ep-curr_idx
        train_idxs.extend(range(curr_idx, curr_idx+stop_idx))
        curr_ts = curr_ts+train_div
        curr_idx = curr_idx+stop_idx
        # Get test indexes
        stop_idx = np.where(h5f['timestamp'][curr_idx:ep] > curr_ts+test_div)[0]
        stop_idx = stop_idx[0] if np.any(stop_idx) else ep-curr_idx
        test_idxs.extend(range(curr_idx, curr_idx+stop_idx))
        curr_ts = curr_ts+test_div
        curr_idx = curr_idx+stop_idx
    # Replace as necessary
    if force and 'train_idxs' in h5f:
        del h5f['train_idxs']
    h5f.create_dataset('train_idxs', data=np.array(train_idxs))
    if force and 'test_idxs' in h5f:
        del h5f['test_idxs']
    h5f.create_dataset('test_idxs', data=np.array(test_idxs))
    return

class MultiHDF5SeqVisualIterator(object):
    def flow(self, h5fs, dataset_keys, indexes_key, batch_size, seq_length=30, shuffle=True, return_time=False, speed_gt=0):
        # Get some constants
        all_data_idxs = []
        dataset_lookup = {h5f:dataset_key for h5f, dataset_key in zip(h5fs, dataset_keys)}
        for h5f in h5fs:
            avail_idxs = set(np.array(h5f[indexes_key]))
            for idx in np.array(h5f[indexes_key]):
                # Check to make sure whole sequence is in the same train/test, and filter on speed
                above_speed = np.all(np.array(h5f['vehicle_speed'][idx:idx+seq_length]) > speed_gt)
                if (idx+seq_length in avail_idxs) and above_speed:
                    all_data_idxs.append((h5f, idx))
        num_examples = len(all_data_idxs)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(all_data_idxs)
        b = 0
        while b < num_batches:
            curr_idxs = all_data_idxs[b*batch_size:(b+1)*batch_size]
            todo_dict = defaultdict(list)
            for (h5f, idx) in curr_idxs:
                todo_dict[h5f].append(idx)
            vids, bY, times = [], [], []
            for h5f, idxs in todo_dict.items():
                vids.extend([np.array(h5f[dataset_lookup[h5f]][curr_idx:curr_idx+seq_length]) for curr_idx in idxs])
                bY.extend([h5f['steering_wheel_angle'][curr_idx:curr_idx+seq_length] for curr_idx in idxs])
                times.extend([h5f['timestamp'][curr_idx:curr_idx+seq_length] for curr_idx in idxs])

            # Add a single-dimensional color channel for grayscale
            vids = np.expand_dims(vids, axis=2).astype('float32')/255.-0.5
            # Get rid of infinity and NaNs which sometimes happens from normalizing
            vids[np.isnan(vids)] = 0.
            vids[np.isinf(vids)] = 0.
            # Turn into a numpy-compatible vector
            bY = np.array(bY)
            if return_time:
                yield [vids.astype('float32'), bY.astype('float32'), np.array(times).astype('float32')]
            else:
                yield [vids.astype('float32'), bY.astype('float32')]
            b += 1

class HDF5SeqVisualIterator(object):
    def flow(self, h5f, dataset_key, indexes_key, batch_size, set_length=30, shuffle=True):
        # Get some constants
        mean_vid, std_vid = h5f[dataset_key + '_mean'], h5f[dataset_key + '_mean']
        data_idxs = np.array(h5f[indexes_key])
        data_idx_set = set(data_idxs)
        allowed_idxs = np.array([data_idx for data_idx in data_idxs if data_idx+set_length in data_idx_set])
        num_examples = len(allowed_idxs)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(allowed_idxs)
        b = 0
        while b < num_batches:
            curr_idxs = list(np.sort(data_idxs[b*batch_size:(b+1)*batch_size]))
            vids = [(np.array(h5f[dataset_key][curr_idx:curr_idx+set_length]) - mean_vid) / std_vid for curr_idx in curr_idxs]
            bY = [h5f['steering_wheel_angle'][curr_idx:curr_idx+set_length] for curr_idx in curr_idxs]

            # Add a single-dimensional color channel for grayscale
            vids = np.expand_dims(vids, axis=2)
            vids[np.isnan(vids)] = 0
            vids[np.isinf(vids)] = 0
            bY = np.array(bY)
            yield [vids.astype('float32'), bY.astype('float32')]
            b += 1

class HDF5VisualIterator(object):
    def flow(self, h5f, dataset_key, indexes_key, batch_size, shuffle=True):
        # Get some constants
        mean_vid, std_vid = h5f[dataset_key + '_mean'], h5f[dataset_key + '_mean']
        data_idxs = np.array(h5f[indexes_key])
        num_examples = len(data_idxs)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_idxs)
        b = 0
        while b < num_batches:
            curr_idxs = list(np.sort(data_idxs[b*batch_size:(b+1)*batch_size]))
            vid = (h5f[dataset_key][curr_idxs] - mean_vid) / std_vid
            bY = h5f['steering_wheel_angle'][curr_idxs]

            # Add a single-dimensional color channel for grayscale
            vid = np.expand_dims(vid, axis=1)
            vid[np.isnan(vid)] = 0.
            vid[np.isinf(vid)] = 0.
            bY = np.expand_dims(bY, axis=1)
            yield [vid.astype('float32'), bY.astype('float32')]
            b += 1

class MultiHDF5VisualIterator(object):
    def flow(self, h5fs, dataset_keys, indexes_key, batch_size, shuffle=True, seperate_dvs_channels=False):
        # Get some constants
        all_data_idxs = []
        dataset_lookup = {h5f:dataset_key for h5f, dataset_key in zip(h5fs, dataset_keys)}
        for h5f in h5fs:
            for idx in np.array(h5f[indexes_key]):
                all_data_idxs.append((h5f, idx))
        num_examples = len(all_data_idxs)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(all_data_idxs)
        b = 0
        while b < num_batches:
            curr_idxs = all_data_idxs[b*batch_size:(b+1)*batch_size]
            todo_dict = defaultdict(list)
            for (h5f, idx) in curr_idxs:
                todo_dict[h5f].append(idx)
            vids = []
            bY = []
            for h5f, idxs in todo_dict.items():
                print(h5f)
                print(idxs)
                vids.extend(h5f[dataset_lookup[h5f]][sorted(idxs)])
                bY.extend(h5f['steering_wheel_angle'][sorted(idxs)])

            # Add a single-dimensional color channel for grayscale
            if seperate_dvs_channels:
                vids = np.array(vids)/255.-0.5
            else:
                vids = np.expand_dims(vids, axis=1).astype('float32')/255.-0.5
            vids[np.isnan(vids)] = 0.
            vids[np.isinf(vids)] = 0.
            bY = np.expand_dims(bY, axis=1)
            yield [vids, bY.astype('float32')]
            b += 1

class MultiHDF5VisualIteratorFederated(object):
    def flow(self, h5fs, dataset_keys, indexes_key, batch_size, shuffle=True, seperate_dvs_channels=False, iid = True, client_id = 0, num_clients = 1, seq_length = 10, speed_gt = 15):
        if iid:
            all_data_idxs = []
            dataset_lookup = {h5f:dataset_key for h5f, dataset_key in zip(h5fs, dataset_keys)}
            count = 0
            for h5f in h5fs:
                avail_idxs = set(np.array(h5f[indexes_key]))
                for idx in np.array(h5f[indexes_key]):
                    if count%num_clients == client_id:
                        above_speed = np.all(np.array(h5f['vehicle_speed'][idx:idx+seq_length]) > speed_gt)
                        if (idx+seq_length in avail_idxs) and above_speed:
                            all_data_idxs.append((h5f, idx))
                    count += 1
                    if count == num_clients:
                        count = 0
        else:
            all_data_idxs = []
            h5f = h5fs[client_id]
            dataset_key = dataset_keys[client_id]
            dataset_lookup = {h5f:dataset_key}
            for idx in np.array(h5f[indexes_key]):
                above_speed = np.all(np.array(h5f['vehicle_speed'][idx:idx+seq_length]) > speed_gt)
                if (idx+seq_length in avail_idxs) and above_speed:
                    all_data_idxs.append((h5f, idx))

        num_examples = len(all_data_idxs)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(all_data_idxs)
        b = 0
        while b < num_batches - 1:
            curr_idxs = all_data_idxs[b*batch_size:(b+1)*batch_size]
            todo_dict = defaultdict(list)
            for (h5f, idx) in curr_idxs:
                todo_dict[h5f].append(idx)
            vids = []
            bY = []
            for h5f, idxs in todo_dict.items():
                vids.extend(h5f[dataset_lookup[h5f]][sorted(idxs)])
                bY.extend(h5f['steering_wheel_angle'][sorted(idxs)])

            # Add a single-dimensional color channel for grayscale
            if seperate_dvs_channels:
                vids = np.array(vids)/255.-0.5
            else:
                vids = np.expand_dims(vids, axis=1).astype('float32')/255.-0.5
            vids[np.isnan(vids)] = 0.
            vids[np.isinf(vids)] = 0.
            bY = np.expand_dims(bY, axis=1)
            yield [vids, bY.astype('float32')]
            b += 1

def resize_int8(frame, size):
    #return imresize(frame, size)
    return img_as_ubyte(resize(frame, size))

def resize_int16(frame, size=(60,80), method='bilinear', climit=[-15,15], seperate_dvs_channels=False, split_timesteps=False, timesteps=10):
    # Assumes data is some small amount around the mean, i.e., DVS event sums
    #return imresize((np.clip(frame, climit[0], climit[1]).astype('float32')+127), size, interp=method).astype('uint8')

    if split_timesteps:
        out_frame = img_as_ubyte(resize(np.clip(frame, climit[0], climit[1]), (timesteps, 2, size[0], size[1])))
    elif seperate_dvs_channels:
        out_frame = img_as_ubyte(resize(np.clip(frame, climit[0], climit[1]), (2, size[0], size[1])))
    else:
        out_frame = img_as_ubyte(resize(np.clip(frame, climit[0], climit[1]), (size[0], size[1])))
    return out_frame

def dvs_to_aps(dvs_image, model, device):
    dvs_batch = dvs_image[None, ...] # Create batch of size 1
    with torch.no_grad():
        aps_batch = model(torch.from_numpy(dvs_batch).to(device=device, dtype=torch.float))
    aps_image = aps_batch[0, 0, ...]
    return aps_image

def resize_data_into_new_key(h5f, key, new_key, new_size, chunk_size=1024, seperate_dvs_channels=False, split_timesteps=False, timesteps = 10):
    chunk_generator = yield_chunker(h5f[key], chunk_size)

    # Set some basics
    dtype = h5f[key].dtype
    row_idx = 0

    if key == 'aps_frame':
        do_resize = resize_int8
    elif key == 'dvs_frame' or key == 'dvs_split' or key == 'dvs_channels':
        do_resize = resize_int16
    else:
        raise AssertionError('Unknown data type')
    # Initialize a resizable dataset to hold the output
    if split_timesteps:
        resized_shape = (chunk_size, timesteps, 2,) + new_size
    elif seperate_dvs_channels:
        resized_shape = (chunk_size, 2,) + new_size
    else:
        resized_shape = (chunk_size,) + new_size
    max_shape = (h5f[key].shape[0],) + resized_shape[1:]

    if new_key in h5f:
        dset = h5f[new_key]
    else:
        dset = h5f.create_dataset(new_key, shape=max_shape, maxshape=max_shape,
                            chunks=resized_shape, dtype='uint8')
    # Write all data out, one chunk at a time
    for chunk in chunk_generator:
        # Operate on the data
        if split_timesteps:
            resized_chunk = np.array([do_resize(frame, new_size, split_timesteps=split_timesteps, timesteps=timesteps) for frame in chunk])
        elif seperate_dvs_channels:
            resized_chunk = np.array([do_resize(frame, new_size, seperate_dvs_channels=seperate_dvs_channels) for frame in chunk])
        else:
            resized_chunk = np.array([do_resize(frame, new_size) for frame in chunk])
        # Write the next chunk
        dset[row_idx:row_idx+chunk.shape[0]] = resized_chunk
        # Increment the row count
        row_idx += chunk.shape[0]

# def pad_datastreams(datastreams, rand_pad=10):
#     num_streams = len(datastreams)
#     max_len = np.max([[len(item) for item in datastream] for datastream in datastreams])
#     max_len = max_len + rand_pad
#     returned_streams = []
#     for datastream in datastreams:
#         num_items = len(datastream)
#         # First dim is batch size, second is length (going to be overwritten to max)
#         size_tuple = datastream[0].shape[1:]
#         data_tensor = np.zeros( (num_items, max_len) + size_tuple, dtype='float32')
#         mask_tensor = np.zeros( (num_items, max_len), dtype='float32')
#         for idx, item in enumerate(datastream):
#             start_offset = np.random.randint(low=0, high=max_len-len(item))
#             data_tensor[idx, start_offset:start_offset+len(item)] = item
#             mask_tensor[idx, start_offset:start_offset+len(item)] = 1.
#         returned_streams.append(data_tensor)
#         returned_streams.append(mask_tensor)
#     return returned_streams
