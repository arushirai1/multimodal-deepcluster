import os, sys
import numpy as np
import cv2
import errno
import pandas as pd
import pickle
def build_paths(print_paths=True):
    root= '/home/mschiappa/data/COIN/'
    dictionary_pickle='coin_howto_overlap_captions.pickle'
    metadata_path = 'coin_howto_overlap_metadata.csv'

    if print_paths:
        print('Video Root Path: %s' % root,
              '\nDictionary Path: %s' % str(dictionary_pickle),
              '\nMetadata Path: %s' % str(metadata_path))

    return root, dictionary_pickle, metadata_path
def grab_frames_by_time(video_file='cPEFskCrdhQ', start=22.49, end=30.44, cliplen=16):
    if not os.path.exists(video_file + '.mp4'):
        print("DNE", video_file)
    cap = cv2.VideoCapture("%s.mp4" % (video_file))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    # step=max(length//desired_frames,2)
    for i in np.linspace(FPS * start, FPS * end, num=cliplen, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if (frame is not None and frame.size != 0):
            frames.append(frame)
    frames = np.array(frames).astype(np.float32)
    return frames

def _balance_targets(metadata_path):
    df = pd.read_csv(metadata_path)
    threshold = 20
    classes = df.groupby(['class']).video_url.nunique().reset_index().sort_values(by='video_url', ascending=False)
    classes = classes[classes['video_url'] > threshold]
    np.sum(classes['video_url'])
    df_test = df[df['class'].isin(classes['class'].values)]
    return df_test

def get_keys(path_to, dictionary_path, metadata_path):
    clip_list = []
    test_list = []
    classes = set()
    with open(dictionary_path, 'rb') as f:
        captions_df = pickle.load(f)
    df = _balance_targets(metadata_path)
    keys = captions_df.keys()

    for key in df['Unnamed: 0'].values:
        class_target = df[df['Unnamed: 0'] == key]['class'].values[0]
        classes.add(class_target)
        key += '.mp4'
        mid = ''
        if os.path.exists(os.path.join(path_to, 'train', key)):
            mid = 'train'
            clip_list.append((os.path.join(path_to,'train', key), class_target))
        elif os.path.exists(os.path.join(path_to, 'val', key)):
            mid = 'val'
            test_list.append((os.path.join(path_to,'val', key), class_target))

    print(clip_list)
    print("Train Length", len(clip_list))
    print("Test Length", len(test_list))

    return clip_list, test_list, classes

def get_text_description(dictionary_path, key):
    with open(dictionary_path, 'rb') as f:
        captions_df = pickle.load(f)
        return ' '.join(list(captions_df[key]['text']))

import random
def rand(count, start, end, spacing):
    l = []
    if(end//spacing < count):
        return None
    for i in range(count):
        while True:
            num = random.randrange(start, end, spacing)
            if num not in l:
                l.append(num)
                break
    return l

#continuous
def create_frames(video_file, desired_frames=64, skip_rate=2, continuous_frames=8):
    if not os.path.exists(video_file):
        print("DNE", video_file)
        return None
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #i = random.randrange(0, length, desired_frames*skip_rate)
    end = length - desired_frames*skip_rate - 1
    if end <= 0:
        i=0
        skip_rate=1
    else:
        i = random.randint(0, end)
    frames = []
    for j in range(0,desired_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i+skip_rate*j)
        success, frame = cap.read()
        if success and (frame is not None and frame.size != 0):
            frame=np.nan_to_num(frame, 0, posinf=255, neginf=0)
            low_mask = frame < 0
            high_mask = frame > 255
            frame[low_mask]=0
            frame[high_mask]=255
            frames.append(frame)
        else:
            if len(frames) is not 0:
                frames.append(frames[-1])
    while len(frames) != desired_frames:
        frames.append(frames[-1])
        print("video file err", video_file)
    frames=np.array(frames).astype(np.float32)
    return frames
'''
def create_frames(video_file, desired_frames=64, continuous_frames=8):
    if not os.path.exists(video_file+'.avi'):
        print("DNE", video_file)
        return None
    times = desired_frames//continuous_frames
    cap = cv2.VideoCapture("%s.avi" %(video_file))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_starts = rand(times,0,length-continuous_frames,continuous_frames)
    frames = []
    print(clip_starts)
    #step=max(length//desired_frames,2)
    for i in clip_starts: #np.linspace(0,length, num=desired_frames, dtype=int):
        for j in range(0, continuous_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i+j)
            success, frame = cap.read()
            if (frame is not None and frame.size != 0):
                #frame = cv2.patchNaNs(frame, 0)
                frames.append(frame)
    frames=np.array(frames).astype(np.float32)
    return frames
'''
'''
def create_frames(video_file, desired_frames=64):
    if not os.path.exists(video_file+'.avi'):
        print("DNE", video_file)
        return None
    cap = cv2.VideoCapture("%s.avi" %(video_file))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    #step=max(length//desired_frames,2)
    for i in np.linspace(0,length, num=desired_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if (frame is not None and frame.size != 0):
            frames.append(frame)
    frames=np.array(frames).astype(np.float32)
    return frames
'''
# Computes Mean and Std Dev, across RGB channels, of all training images in a Dataset & returns averages
# Set Pytorch Transforms to None for this function
def calc_mean_and_std(dataset):
    mean = np.zeros((3,1))
    std = np.zeros((3,1))
    print('==> Computing mean and std...')
    for img in dataset:
        scaled_img = np.array(img[0])/255
        mean_tmp, std_tmp = cv2.meanStdDev(scaled_img)
        mean += mean_tmp
        std += std_tmp
    mean = mean/len(dataset)
    std = std/len(dataset)

    return mean, std


def cv2_imshow(img_path, wait_key=0, window_name='Test'):
    img = cv2.imread(img_path)
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()
