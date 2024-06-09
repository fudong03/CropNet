import h5py
import torch
import os
import numpy as np
import json
from einops import rearrange


class Sentinel2Imagery(torch.utils.data.Dataset):

    def __init__(self, base_dir, config_file, transform=None):
        """
            Dataset for Sentinel-2 Imagery

        :param base_dir: the root directory for CropNet dataset, e.g., /mnt/data/CropNet
        :param config_file: the path for JSON configuration file
        :param transform:
        """
        self.fips_codes = []
        self.years = []
        self.file_paths = []
        self.transform = transform

        data = json.load(open(config_file))
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            tmp_path = []
            relative_path_list = obj["data"]["sentinel"]
            for relative_path in relative_path_list:
                tmp_path.append(os.path.join(base_dir, relative_path))
            self.file_paths.append(tmp_path)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year = self.fips_codes[index], self.years[index]
        file_paths = self.file_paths[index]

        temporal_list = []

        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                groups = hf[fips_code]
                for d in groups.keys():
                    grids = groups[d]["data"]
                    grids = torch.from_numpy(np.asarray(grids))
                    temporal_list.append(grids)
                hf.close()

        x = torch.stack(temporal_list)
        x = x.to(torch.float32)
        x = rearrange(x, 't g h w c -> t g c h w')

        if self.transform:
            t, g, _, _, _ = x.shape
            x = rearrange(x, 't g c h w -> (t g) c h w')
            x = self.transform(x)
            x = rearrange(x, '(t g) c h w -> t g c h w', t=t, g=g)

        return x, fips_code, year