import torch
from torch.utils.data import Dataset
import os
import json
import pandas as pd


class USDACropDataset(torch.utils.data.Dataset):

    def __init__(self, base_dir, config_file, crop_type="Soybeans"):
        """
            Dataset for the USDA Crop Dataset

        :param base_dir: the root directory for CropNet dataset, e.g., /mnt/data/CropNet
        :param config_file: the path for JSON configuration file
        :param crop_type: the crop type for use. Choices: ["Corn", "Cotton", "Soybeans", "Winter Wheat"]
        """

        all_crop_types = ["Corn", "Cotton", "Soybeans", "Winter Wheat"]

        # validate the crop type
        assert crop_type in all_crop_types, "Cannot find a crop type named {} in the USDA Crop Dataset."

        self.crop_type = crop_type

        if crop_type == "Cotton":
            column_names = ['PRODUCTION, MEASURED IN 480 LB BALES', 'YIELD, MEASURED IN BU / ACRE']
        else:
            column_names = ['PRODUCTION, MEASURED IN BU', 'YIELD, MEASURED IN BU / ACRE']

        self.column_names = column_names

        self.fips_codes = []
        self.years = []
        self.state_ansi = []
        self.county_ansi = []
        self.file_paths = []

        data = json.load(open(config_file))
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])
            self.state_ansi.append(obj["state_ansi"])
            self.county_ansi.append(obj["county_ansi"])

            relative_path = obj["data"]["USDA"]
            self.file_paths.append(os.path.join(base_dir, relative_path))

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year = self.fips_codes[index], self.years[index]
        state_ansi, county_ansi = self.state_ansi[index], self.county_ansi[index]

        file_path = self.file_paths[index]
        df = pd.read_csv(file_path)

        # convert state_ansi and county_ansi to string with leading zeros
        df['state_ansi'] = df['state_ansi'].astype(str).str.zfill(2)
        df['county_ansi'] = df['county_ansi'].astype(str).str.zfill(3)

        df = df[(df["state_ansi"] == state_ansi) & (df["county_ansi"] == county_ansi)]

        df = df[self.column_names]

        x = torch.from_numpy(df.values)
        x = x.to(torch.float32)
        x = torch.log(torch.flatten(x, start_dim=0))

        return x, fips_code, year
