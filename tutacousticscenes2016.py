"""TUT Acoustic scenes 2016 dataset.
"""
import os
import sys
import numpy as np
import zipfile
import librosa
from tqdm import tqdm
from keras.utils.data_utils import get_file


def __load_files(fileset, fileset_path, labels, n=None):
    # source_sample_rate = 44100
    # source_len = 30 * 44100 + 1 # 30 seconds audio files
    load_sample_rate = 16000
    n_samples_per_file = 30 * load_sample_rate + 1

    if n is None:
        n = len(fileset)

    print("Loading wav files to " + fileset_path)

    # Allocate return buffers
    x = np.empty([n, n_samples_per_file], dtype='float32', order='C')
    y = np.empty(n, dtype='int16')

    for i in tqdm(range(n)):
        meta = fileset[i]
        wav_filename = os.path.join(fileset_path, meta[0])
        label = meta[1]
        # Load audio file as a floating point time series.
        # Data is converted from:
        # int, 44.1kHz stereo to
        # normalized float, 16kHz, mono
        # @note: resampling takes time!! (~1.4s per call)
        x[i], _ = librosa.load(wav_filename, sr=16000, mono=True, dtype=np.float32)
        # x[i], _  = librosa.load(wav_filename, sr=44100, mono=True, dtype=np.float32) # no resampling
        y[i] = labels[label]

    return (x, y)


def __content_valid(infolist, destination):
    for file in infolist:
        # ignore directories
        if file.filename.endswith('/'):
            continue
        # check filesize match for speed (or check CRC)
        path = os.path.join(destination, file.filename)
        try:
            size = os.path.getsize(path)
            if size != file.file_size:
                return False
        except OSError:
            return False

    return True


def load_data():
    """Loads the TUT Acoustic scenes 2016 dataset.
    # Returns
    Tuple of Numpy arrays: `(x_dev, y_dev), (x_eval, y_eval)`.
    """
    dirname = os.path.join('/content/sample_data/datasets', 'TUT-acoustic-scenes-2016')
    datadir_base = '/content/sample_data/datasets'  # current directory
    datadir = os.path.join(datadir_base, dirname)
    dev_fileset_path = os.path.join(datadir, 'TUT-acoustic-scenes-2016-development')
    eval_fileset_path = os.path.join(datadir, 'TUT-acoustic-scenes-2016-evaluation')
    npz_dev_filepath = os.path.join(datadir, 'development_set.npz')
    npz_eval_filepath = os.path.join(datadir, 'evaluation_set.npz')
    dev_meta_filepath = os.path.join(dev_fileset_path, 'meta.txt')
    eval_meta_filepath = os.path.join(eval_fileset_path, 'meta.txt')

    # Development dataset (7.5GB)
    dev_base = '/content/sample_data/datasets'
    dev_files = [
        '/content/sample_data/datasets/TUT-acoustic-scenes-2016/TUT-acoustic-scenes-2016-development/audio'
        # 'TUT-acoustic-scenes-2016-development.audio.1.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.2.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.3.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.4.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.5.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.6.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.7.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-development.audio.8.zip',  # 528.5MB
        # 'TUT-acoustic-scenes-2016-development.doc.zip',      # 69.7KB
        # 'TUT-acoustic-scenes-2016-development.error.zip',    # 1.3KB
        # 'TUT-acoustic-scenes-2016-development.meta.zip'      # 28.8KB
    ]

    # Evaluation dataset (2.5 GB)
    eval_base = '/content/sample_data/datasets'
    eval_files = [
        '/content/sample_data/datasets/TUT-acoustic-scenes-2016/TUT-acoustic-scenes-2016-evaluation/audio'
        # 'TUT-acoustic-scenes-2016-evaluation.audio.1.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-evaluation.audio.2.zip',  # 1.1GB
        # 'TUT-acoustic-scenes-2016-evaluation.audio.3.zip',  # 538.9MB
        # 'TUT-acoustic-scenes-2016-evaluation.doc.zip',      # 69.2KB
        # 'TUT-acoustic-scenes-2016-evaluation.meta.zip'      # 6.0KB
    ]

    # 3 classes : 0 indoor, 1 outdoor, 2 vehicle
    # http://www.cs.tut.fi/sgn/arg/dcase2016/acoustic-scenes
    labels = {
        'backward':  0,  # indoor
        'forward':   1,  # indoor
        'stop':      2,  # indoor
        # 'metro_station':    0,  # indoor
        # 'office':           0,  # indoor
        # 'home':             0,  # indoor
        # 'city_center':      1,  # outdoor
        # 'forest_path':      1,  # outdoor
        # 'beach':            1,  # outdoor
        # 'residential_area': 1,  # outdoor
        # 'park':             1,  # outdoor
        # 'car':              2,  # vehicle
        # 'bus':              2,  # vehicle
        # 'train':            2,  # vehicle
        # 'tram':             2,  # vehicle
    }

    # Check if data has already been loaded into NumPy or load from .wav files
    # warning: parameter n (number of loaded files) is not considered
    if os.path.exists(npz_dev_filepath):
        with np.load(npz_dev_filepath) as npzfile:
            x_dev = npzfile['arr_0']
            y_dev = npzfile['arr_1']
            print("Loaded development set from cache")
            dev_set_loaded = 1
    else:
        dev_set_loaded = 0

    if os.path.exists(npz_eval_filepath):
        with np.load(npz_eval_filepath) as npzfile:
            x_eval = npzfile['arr_0']
            y_eval = npzfile['arr_1']
            print("Loaded evaluation set from cache")
            eval_set_loaded = 1
    else:
        eval_set_loaded = 0

    # # Download
    # if not dev_set_loaded:
    #     dev_paths = []
    #     for fname in dev_files:
    #         dev_paths.append(get_file(fname,
    #                             origin=os.path.join(dev_base, fname),
    #                             cache_subdir=dirname,
    #                             cache_dir=datadir_base))
    # if not eval_set_loaded:
    #     eval_paths = []
    #     for fname in eval_files:
    #         eval_paths.append(get_file(fname,
    #                             origin=os.path.join(eval_base, fname),
    #                             cache_subdir=dirname,
    #                             cache_dir=datadir_base))

    # # Extract development set
    # if not dev_set_loaded:
    #     for fname in dev_paths:
    #         with zipfile.ZipFile(fname, 'r') as z:
    #             infolist = z.infolist()
    #             # Check if already extracted
    #             if __content_valid(infolist, datadir) is True:
    #                 print("Extracted dataset found for " + fname)
    #                 sys.stdout.flush()
    #             else:
    #                 print("Extracting " + fname)
    #                 sys.stdout.flush()
    #                 z.extractall(datadir)

    # # Extract evaluation set
    # if not eval_set_loaded:
    #     for fname in eval_paths:
    #         with zipfile.ZipFile(fname, 'r') as z:
    #             infolist = z.infolist()
    #             # Check if already extracted
    #             if __content_valid(infolist, datadir) is True:
    #                 print("Extracted dataset found for " + fname)
    #                 sys.stdout.flush()
    #             else:
    #                 print("Extracting " + fname)
    #                 sys.stdout.flush()
    #                 z.extractall(datadir)

    # load dataset meta data, csv-format, [audio file (string)][tab][scene label (string)]
    if not dev_set_loaded:
        dev_fileset = np.loadtxt(dev_meta_filepath, dtype=str)
        # Load wav data
        x_dev, y_dev = __load_files(dev_fileset, dev_fileset_path, labels)
        # Save data
        np.savez(npz_dev_filepath, x_dev, y_dev)

    if not eval_set_loaded:
        eval_fileset = np.loadtxt(eval_meta_filepath, dtype=str)
        # Load wav data
        x_eval, y_eval = __load_files(eval_fileset, eval_fileset_path, labels)
        # Save data
        np.savez(npz_eval_filepath, x_eval, y_eval)

    return (x_dev, y_dev), (x_eval, y_eval)


