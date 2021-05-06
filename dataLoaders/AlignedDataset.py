import os
from utils.DatasetUtils import get_params, get_transform
from PIL import Image

from utils import is_img


class AlignedDataset:
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, cfg, stage):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        m_cfg = cfg.copy()
        m_cfg.update(cfg['dataset'])
        m_cfg.update(cfg['dataset'][stage])
        self.cfg = m_cfg
        self.dir_AB = os.path.join(m_cfg['dataroot'], stage)  # get the image directory
        self.AB_paths = []
        self.load_samples() # get image paths

        self.load_size = m_cfg.get('load_size', 512)
        self.crop_size = m_cfg.get('crop_size', 384)

        assert(self.load_size >= self.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = m_cfg['output_nc'] if m_cfg['direction'] == 'BtoA' else m_cfg['input_nc']
        self.output_nc = m_cfg['input_nc'] if m_cfg['direction'] == 'BtoA' else m_cfg['output_nc']

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.cfg, A.size)
        A_transform = get_transform(self.cfg, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.cfg, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def load_samples(self):
        for f in os.listdir(self.dir_AB)[:self.cfg['max_dataset_size']]:
            if is_img(f):
                self.AB_paths.append(os.path.join(self.dir_AB, f))
