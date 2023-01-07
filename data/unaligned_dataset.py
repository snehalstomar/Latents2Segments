import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    
    def __init__(self, opt):
        
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "testA")
            self.dir_B = os.path.join(opt.dataroot, "testB")

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        random.Random(0).shuffle(self.A_paths)
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        random.Random(0).shuffle(self.B_paths)
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt, grayscale=False)
        self.transform_B = get_transform(self.opt, grayscale=False)
        self.B_indices = list(range(self.B_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if index == 0 and self.opt.isTrain:
            random.shuffle(self.B_indices)
        index_B = self.B_indices[index % self.B_size]
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'real_A': A, 'real_B': B, 'path_A': A_path, 'path_B': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
