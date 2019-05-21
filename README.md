This is a utility intended to be used to load videos in as batches to an arbitrary PyTorch model. It is based off of the [pynvvl library](https://github.com/mitmul/pynvvl), so you will need to install this. Also, if you want to use the resize option with GPU support, you will need to install [NVIDIA's video codec sdk](https://developer.nvidia.com/nvidia-video-codec-sdk). If you don't want to do this, you can pull my docker image, **michaelteti/video**, which as everything already installed (highly recommended). Below are the arguments to the video loader, and you can find an example of how to use it during training if you go to [example.py](https://github.com/MichaelTeti/video_loader/blob/master/example.py).

```
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
```

You must have a file called **params.json** in the directory you run the main script from. This **params.json** file should look something like this (still working on model params):

```
{
    "model_params": {
        "test": "test"
    },
    "video_loading_params": {
        "crop_wd": 256,
        "horiz_flip": true,
        "save_fnames": false,
        "downsample_t": 2,
        "resize_vids": false,
        "n_frames": 5,
        "crop_ht": 256,
        "batch_size": 32,
        "normalize": true,
        "device_id": 0,
        "center_crop": true,
        "exclude_dirs": ["raw", "images"],
        "resize_dir": "/dataset/dataset_mod/originals_resized",
        "data_path": "/dataset/dataset_mod/originals_resized",
        "exclude_files": null
    }
}
```
