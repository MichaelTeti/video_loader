from pynvvl import NVVLVideoLoader as NVVL
import os
import numpy as np
from imageio import get_reader, get_writer
from progressbar import ProgressBar
import csv
from torch.utils.dlpack import from_dlpack
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import subprocess
import json
import multiprocessing
import sys


class VideoUtil:
    def __init__(self, data_dir, exclude_dirs=None, exclude_files=None,
                 horiz_flip=False, normalize=True, device_id=0,
                 save_fnames=False, n_frames=1, crop_ht=224, crop_wd=224,
                 center_crop=True, downsample_t=1, batch_size=32,
                 resize_vids=False, resize_dir=None):

        '''
        A class to handle loading and processing video data using the PYNVVL
        library (https://github.com/mitmul/pynvvl).

        Args:
            data_dir (str): The path to the folder containing either video files
                or other folders containing video files. os.walk is used to
                travel down starting at this path.
            exclude_dirs (list of strings): Whether to exclude certain folder
                names when walking down the file system. Can be full names or
                part of a name. Default is None, which will not exclude any
                folder.
            exclude_files (list of strings): Whether to exclude any file names
                when traversing down the file system. Can be full names or part of
                a name, such as an extension. Default is None, which will not
                exclude any files based on name.
            horiz_flip (bool): Whether to randomly flip some frames across the
                y-axis. Default is False.
            normalize (bool): Whether or not to normalize each video's frames
                so that their pixel values are in the range [0, 1]. Default is
                True.
            device_id (int): Indicates which GPU device to load the videos onto.
                Default is device 0.
            save_fnames (bool): True to save filenames in a .csv file for faster
                loading or False to use os.walk() to find them again. Default
                False.
            n_frames (int >= 1): Number of consecutive video frames to stack
                together to construct a single training example. Default is 1,
                which means a single frame.
            crop_ht (int >= 0): height to crop each image to. Default 224. If
                crop_ht is 0, no cropping will be performed.
            crop_wd (int): width to crop each image to. Default 224. If crop_wd
                is 0, no cropping will be performed.
            center_crop (bool): True to crop at center of frames, False to
                perform random crops. Default False.
            downsample_t (int >= 1): How much to downsample in time. Default is
                1, which means no downsampling.
            batch_size (int >= 0): Batch size for each training iteration.
                Default is 32 examples / iteration.
            resize_vids (bool): True to resize videos to a smaller size first,
                False to keep them as the original sizes. Default False.
            resize_dir (str): Directory to put the resized videos in if
                resize_vids is True. Default is None.
        '''

        self.data_dir = data_dir
        self.exclude_dirs = exclude_dirs
        self.exclude_files = exclude_files
        self.save_fnames = save_fnames
        self.filenames, self.classes = self.get_file_names()

        self.horiz_flip = horiz_flip
        self.normalize = normalize
        self.batch_size = batch_size
        self.batch_num = 0
        self.n_batches = 0

        assert(n_frames >= 1), 'n_framestack must be >= 1'
        self.n_frames = n_frames
        if self.n_frames > 1:
            self.frames_need = self.batch_size + self.n_frames // 2 * 2
        else:
            self.frames_need = self.batch_size

        assert(crop_ht >= 0), 'crop_ht must be >= 0. Enter a value from this range.'
        self.crop_ht = crop_ht

        assert(crop_wd >= 0), 'crop_wd must be >= 0. Enter a value from this range.'
        self.crop_wd = crop_wd

        self.center_crop = center_crop
        self.downsample_t = downsample_t

        if resize_vids:
            assert(resize_dir is not None), 'Provide resize_dir if resize_vids is True.'
            assert(resize_dir is not data_dir), 'resize_dir must not equal data_dir.'
            self.resize_dir = resize_dir
            if not os.path.isdir(resize_dir):
                os.mkdir(resize_dir)

            self.resize_and_save()
            self.modify_params('data_path', resize_dir)
            self.modify_params('resize_vids', False)
            sys.exit(0)





    def modify_params(self, param, new_val):
        with open('params.json', 'r') as jsonfile:
            params = json.load(jsonfile)

            for k, v in params.items():
                for k2, v2 in v.items():
                    if k2 == param:
                        params[k][k2] = new_val

        with open('params.json', 'w') as jsonfile:
            json.dump(params, jsonfile, indent=4)






    def get_file_names(self):
        filenames = []
        if 'filenames.csv' in os.listdir():
            with open('filenames.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                for row in reader:
                    filenames.append(row[0])

        else:
            for root, dirs, files in os.walk(self.data_dir):
                cont = False
                if self.exclude_dirs is not None:
                    for exclude_dir in self.exclude_dirs:
                        if exclude_dir in root:
                            cont = True

                if cont:
                    continue

                for file in files:
                    cont2 = False
                    if self.exclude_files is not None:
                        for exclude_file in self.exclude_files:
                            if exclude_file in file:
                                cont2 = True

                    if cont2:
                        continue

                    filenames.append(os.path.join(root, file))

                    if self.save_fnames:
                        with open('filenames.csv', 'a') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter=',')
                            csvwriter.writerow([os.path.join(root, file)])

        classes = set([os.path.split(os.path.split(f)[0])[1] for f in filenames])
        print('[INFO] Found {} valid files from {} classes.'.format(len(filenames),
                                                                    len(classes)))
        return filenames, list(classes)





    def scale_vids(self, vid_size):
        vid_size_norm = np.amin(np.asarray(vid_size)) / 360
        return np.round(np.exp(-(.2 * vid_size_norm-.534)) - .4, 2)





    def resize(self, fnames):
        for fname in fnames:
            if os.path.split(fname)[1] not in os.listdir(self.resize_dir):
                vid = get_reader(fname)
                vid_size = vid.get_meta_data()['size']
                n = len(list(vid))
                dur = vid.get_meta_data()['duration']
                fps = vid.get_meta_data()['fps']
                scale_const = self.scale_vids(vid_size)
                w = int(vid_size[0] * scale_const)
                h = int(vid_size[1] * scale_const)
                subprocess.call(
                    ['ffmpeg',
                    '-hwaccel', 'cuvid',
                    '-c:v', 'h264_cuvid',
                    '-resize', '{}x{}'.format(w, h),
                    '-i', fname,
                    '-vf', 'scale_npp=format=yuv420p,hwdownload,format=yuv420p',
                    '-r', str(fps),
                    '-vframes', str(n),
                    '-t', str(dur),
                    '-loglevel', 'quiet',
                    '-y', os.path.join(self.resize_dir, os.path.split(fname)[1])])




    def resize_and_save(self):
        print('[INFO] Resizing Videos and saving at {}'.format(self.resize_dir))
        n_cpus = int(multiprocessing.cpu_count() * .75)
        files_per_worker = len(self.filenames) // n_cpus
        workers = {}

        for cpu in range(n_cpus):
            f_index = cpu * files_per_worker
            worker_names = self.filenames[f_index:f_index+files_per_worker]
            workers['worker {}'.format(cpu+1)] = multiprocessing.Process(target=self.resize,
                                                                         args=(worker_names, ))
            workers['worker {}'.format(cpu+1)].start()

        for worker in workers.keys():
            workers[worker].join()





    def crop(self, frames):
        n, c, h, w = frames.shape

        if self.crop_ht > 0:
            if self.center_crop:
                h_start = h // 2 - self.crop_ht // 2
                h_end = h_start + self.crop_ht
            else:
                h_start = np.random.randint(0, frames.shape[2]-self.crop_ht-1, 1)[0]
                h_end = h_start + self.crop_ht
        else:
            h_start, h_end = 0, frames.shape[2]

        if self.crop_wd > 0:
            if self.center_crop:
                w_start = w // 2 - self.crop_wd // 2
                w_end = w_start + self.crop_wd
            else:
                w_start = np.random.randint(0, frames.shape[3]-self.crop_wd-1, 1)[0]
                w_end = w_start + self.crop_wd
        else:
            w_start, w_end = 0, frames.shape[3]

        return frames[..., h_start:h_end, w_start:w_end]





    def load_vid(self):
        vid_ind = np.random.randint(0, len(self.filenames), 1)[0]
        vid_fname = self.filenames[vid_ind]
        label = [i for i in range(len(self.classes)) if self.classes[i] in vid_fname][0]

        try:
            loader = NVVL(device_id=0, log_level='error')
            vid = loader.read_sequence(vid_fname,
                                      horiz_flip=self.horiz_flip,
                                      normalized=self.normalize)
        except:
            loader = NVVL(device_id=0, log_level='error')
            vid = loader.read_sequence(vid_fname,
                                      horiz_flip=self.horiz_flip,
                                      normalized=self.normalize,
                                      count=200)



        vid = from_dlpack(vid.toDlpack())
        if self.downsample_t > 1:
            vid = vid[::self.downsample_t, ...]

        self.batch_num = 0
        self.n_batches = vid.shape[0] // self.frames_need

        return self.crop(vid), label





    def make_gifs(self, frames):
        n, c, h, w = frames.shape
        frames = frames.unfold(0, self.n_frames, 1)
        frames = frames.permute((0, 1, 4, 2, 3)).contiguous()
        return frames.view(n-self.n_frames//2*2, c*self.n_frames, h, w)




    def get_batch(self, frames):
        idx = self.batch_num * self.frames_need
        self.batch_num += 1
        frames = frames[idx:idx+self.frames_need, ...]
        if self.n_frames > 1:
            frames = self.make_gifs(frames)

        return frames
