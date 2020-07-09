import os, sys, select
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from Utils import build_paths, get_keys, create_frames, get_text_description


class COIN(Dataset):
    """
    Args:
        root (string): Path to video directory
        dictionary_pickle (string): Path to train or test split (.pickle)
        metadata_path (string):Path to csv file.

        method (string): ['text_only', 'joint', 'video_only']
        clip_len (int): Number of frames per sample, i.e. depth of Model input.
        train (bool): Training vs. Testing model. Default is True
        do_crop (bool): Crop or not, no cropping keeps width size at 224. Default True
    """

    def __init__(self, root, dictionary_pickle, metadata_path, train, clip_len=16, method='text_only', do_crop=True):

        self.root = root
        self.method = method
        self.dictionary_pickle = dictionary_pickle
        self.metadata_path = metadata_path
        self.train = train
        self.clip_len = clip_len

        clip_list, test_list, classes = get_keys(root, dictionary_pickle, metadata_path)
        self.class_dict = self.read_class_ind(classes)
        if train:
            self.paths = clip_list
        else:
            self.paths = test_list
        self.data_list = self.build_data_list()

        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112
        self.do_crop = do_crop

    # Reads .txt file w/ each line formatted as "1 ApplyEyeMakeup" and returns dictionary {'ApplyEyeMakeup': 0, ...}
    def read_class_ind(self, classes):
        class_dict = {}

        for label, class_name_key in enumerate(classes):
            class_dict[class_name_key] = int(label)
        print(class_dict)
        return class_dict

    def build_data_list(self):
        paths = self.paths
        class_dict = self.class_dict
        data_list = []
        for vid_dir, class_name in paths:
            label = np.array(class_dict[class_name], dtype=int)
            data_list.append((vid_dir, label, self.clip_len, class_name)) #(key, class_target)

        return data_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        vid_dir, label, frame_count, class_name = self.data_list[index]
        buffer = 0 #np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        if 'text_only' is not self.method:
            buffer = self.load_frames(vid_dir, frame_count)
            if self.do_crop:
                buffer = self.spatial_crop(buffer, self.crop_size)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
        key=(vid_dir.split('/')[-1]).split('.')[0]
        text = get_text_description(self.dictionary_pickle, key)
        return buffer, label, text


    def load_frames(self, vid_dir, frame_count):
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        frames = create_frames(vid_dir, self.clip_len)
        for i, frame in enumerate(frames):
            try:
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            except:
                print('The image %s is potentially corrupt!\nDo you wish to proceed? [y/n]\n' % vid_dir)
                response, _, _ = select.select([sys.stdin], [], [], 15)
                if response == 'n':
                    sys.exit()
                else:
                    frame = np.zeros((buffer.shape[1:]))

            frame = np.array(frame).astype(np.float32)
            buffer[i] = frame

        return buffer


    @staticmethod
    def spatial_crop(buffer, crop_size):
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        # Spatial crop is performed on the entire array, so each frame is cropped in the same location.
        buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]

        return buffer


    @staticmethod
    def normalize(buffer):
        for i, frame in enumerate(buffer):
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            with np.errstate(all="raise"):
                frame -= np.array([[[90.0, 98.0, 102.0]]])  # BGR means
            buffer[i] = frame

        return buffer


    @staticmethod
    def to_tensor(buffer):
        buffer = buffer.transpose((3, 0, 1, 2))
        return torch.from_numpy(buffer)

if __name__ == '__main__':

    batch_size = 40
    root, dictionary_pickle, metadata_path = build_paths()
    trainset = COIN(root, dictionary_pickle, metadata_path, train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    for i, (batch, labels) in enumerate(trainloader):
        labels = np.array(labels)
        print(batch.shape)
