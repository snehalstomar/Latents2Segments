import random
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

class ImageFolderDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.dir_A = opt.dataroot
        self.dir_A_full = self.dir_A + '/full'
        self.dir_A_feat_1 = self.dir_A + '/hair'
        self.dir_A_feat_2 = self.dir_A + '/skin'
        self.dir_A_feat_3 = self.dir_A + '/nose'
        self.dir_A_feat_4 = self.dir_A + '/eyes'
        self.dir_A_feat_5 = self.dir_A + '/lips'
        
        # self.A_paths = sorted(make_dataset(self.dir_A))
        # self.A_size = len(self.A_paths)
        self.transform_A = get_transform(self.opt, grayscale=False)

        self.A_paths_full = sorted(make_dataset(self.dir_A_full))
        self.A_size_full = len(self.A_paths_full)
        self.transform_A_full = get_transform(self.opt, grayscale=False)

        self.A_paths_feat_1 = sorted(make_dataset(self.dir_A_feat_1))
        self.A_size_feat_1 = len(self.A_paths_feat_1)
        self.transform_A_feat_1 = get_transform(self.opt, grayscale=False)    
        
        self.A_paths_feat_2 = sorted(make_dataset(self.dir_A_feat_2))
        self.A_size_feat_2 = len(self.A_paths_feat_2)
        self.transform_A_feat_2 = get_transform(self.opt, grayscale=False)

        self.A_paths_feat_3 = sorted(make_dataset(self.dir_A_feat_3))
        self.A_size_feat_3 = len(self.A_paths_feat_3)
        self.transform_A_feat_3 = get_transform(self.opt, grayscale=False)

        self.A_paths_feat_4 = sorted(make_dataset(self.dir_A_feat_4))
        self.A_size_feat_4 = len(self.A_paths_feat_4)
        self.transform_A_feat_4 = get_transform(self.opt, grayscale=False)

        self.A_paths_feat_5 = sorted(make_dataset(self.dir_A_feat_5))
        self.A_size_feat_5 = len(self.A_paths_feat_5)
        self.transform_A_feat_5 = get_transform(self.opt, grayscale=False)
    
    def getitem_by_path(self, A_path):
        try:
            A_img = Image.open(A_path).convert('RGB')    
        except OSError as err:
            print(err)
            return self.__getitem__(random.randint(0, len(self) - 1))

        # apply image transformation
        A = self.transform_A(A_img)
        return {'real_A': A, 'path_A': A_path}

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size_full]
        A_path_full = self.A_paths_full[index % self.A_size_full]                            
        A_path_feat_1 = self.A_paths_feat_1[index % self.A_size_feat_1]
        A_path_feat_2 = self.A_paths_feat_2[index % self.A_size_feat_2]
        A_path_feat_3 = self.A_paths_feat_3[index % self.A_size_feat_3]
        A_path_feat_4 = self.A_paths_feat_4[index % self.A_size_feat_4]
        A_path_feat_5 = self.A_paths_feat_5[index % self.A_size_feat_5]

        #return self.getitem_by_path(A_path)
        
        A_full = self.getitem_by_path(A_path_full)
        A_feat_1 = self.getitem_by_path(A_path_feat_1)
        A_feat_2 = self.getitem_by_path(A_path_feat_2)
        A_feat_3 = self.getitem_by_path(A_path_feat_3)
        A_feat_4 = self.getitem_by_path(A_path_feat_4)
        A_feat_5 = self.getitem_by_path(A_path_feat_5)

        # item_dict = {'real_A_full': A_full['real_A'], 'path_A_full': A_full['path_A'],
        #             'real_A_feat_1': A_feat_1['real_A'], 'path_A_feat_1': A_feat_1['path_A'],
        #             'real_A_feat_2': A_feat_2['real_A'], 'path_A_feat_2': A_feat_2['path_A'],
        #             'real_A_feat_3': A_feat_2['real_A'], 'path_A_feat_3': A_feat_2['path_A'],
        #             'real_A_feat_4': A_feat_2['real_A'], 'path_A_feat_4': A_feat_2['path_A'],
        #             'real_A_feat_5': A_feat_2['real_A'], 'path_A_feat_5': A_feat_2['path_A']}

        
        item_dict = {'real_A_full': A_full['real_A'], 'path_A_full': A_full['path_A'],
                    'real_A_feat_1': A_feat_1['real_A'], 'path_A_feat_1': A_feat_1['path_A'],
                    'real_A_feat_2': A_feat_2['real_A'], 'path_A_feat_2': A_feat_2['path_A'],
                    'real_A_feat_3': A_feat_3['real_A'], 'path_A_feat_3': A_feat_3['path_A'],
                    'real_A_feat_4': A_feat_4['real_A'], 'path_A_feat_4': A_feat_4['path_A'],
                    'real_A_feat_5': A_feat_5['real_A'], 'path_A_feat_5': A_feat_5['path_A']}                                        
        return item_dict            
            
    def __len__(self):
        return self.A_size_full