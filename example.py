import json
from video import *

n_vid_iters = 10 # number of videos to go through during training



assert('params.json' in os.listdir()), "Can't proceed without params.json file."
with open('params.json', 'r') as jsonfile:
    params = json.load(jsonfile)

load_params = params['video_loading_params']


if __name__ == '__main__':
    vd = VideoUtil(load_params['data_path'],
                   exclude_dirs=load_params['exclude_dirs'],
                   exclude_files=load_params['exclude_files'],
                   horiz_flip=load_params['horiz_flip'],
                   normalize=load_params['normalize'],
                   device_id=load_params['device_id'],
                   save_fnames=load_params['save_fnames'],
                   n_frames=load_params['n_frames'],
                   crop_ht=load_params['crop_ht'],
                   crop_wd=load_params['crop_wd'],
                   center_crop=load_params['center_crop'],
                   downsample_t=load_params['downsample_t'],
                   batch_size=load_params['batch_size'],
                   resize_vids=load_params['resize_vids'],
                   resize_dir=load_params['resize_dir'])


    for iter in range(n_vid_iters):
        vid, label = vd.load_vid()
        
        for _ in range(vd.n_batches):
            frames = vd.get_batch(vid)
            print(frames.shape, label)
