import torch
import os
import json
import pandas as pd
from sklearn import preprocessing
from einops import rearrange


class HRRRComputedDataset(torch.utils.data.Dataset):

    def __init__(self, base_dir, config_file, column_names=None):
        """
            Dataset for the HRRR Computed Dataset
        
        :param base_dir: the root directory for CropNet dataset, e.g., /mnt/data/CropNet
        :param config_file: the path for JSON configuration file
        :param column_names: a list of whether parameters for use, i.e., a list of column names in CSV files 
        """

        all_cols = [
            'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
            'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)',
            'Wind Speed (m s**-1)', 'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
            'Downward Shortwave Radiation Flux (W m**-2)', 'Vapor Pressure Deficit (kPa)'
        ]

        # check whether the required weather parameters are valid
        if column_names:
            for col in column_names:
                assert col in all_cols, "Cannot find a weather parameter named {}".format(col)

            self.column_names = column_names
        else:
            default_cols = [
                'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
                'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)',
                'Wind Speed (m s**-1)', 'Downward Shortwave Radiation Flux (W m**-2)',
                'Vapor Pressure Deficit (kPa)'
            ]
            self.column_names = default_cols

        # only consider the first 28 days for addressing different days in each month
        self.day_range = [i + 1 for i in range(28)]

        data = json.load(open(config_file))
        self.fips_codes = []
        self.years = []
        self.short_term_file_path = []
        self.long_term_file_path = []
        self.scaler = preprocessing.StandardScaler()

        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            short_term = []
            for file_path in obj["data"]["HRRR"]["short_term"]:
                short_term.append(os.path.join(base_dir, file_path))

            long_term = []
            for file_paths in obj["data"]["HRRR"]["long_term"]:
                tmp_long_term = []
                for file_path in file_paths:
                    tmp_long_term.append(os.path.join(base_dir, file_path))
                long_term.append(tmp_long_term)

            self.short_term_file_path.append(short_term)
            self.long_term_file_path.append(long_term)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year, = self.fips_codes[index], self.years[index]

        short_term_file_paths = self.short_term_file_path[index]
        x_short = self.get_short_term_val(fips_code, short_term_file_paths)

        long_term_file_paths = self.long_term_file_path[index]
        x_long = self.get_long_term_val(fips_code, long_term_file_paths)

        # convert type
        x_short = x_short.to(torch.float32)
        x_long = x_long.to(torch.float32)

        return x_short, x_long, fips_code, year

    def get_short_term_val(self, fips_code, file_paths):
        """
            Return the daily weather parameters
        :param fips_code: the unique FIPS code for on county
        :param file_paths: the file paths for CSV files
        :return:
        """

        df_list = []
        for file_path in file_paths:
            tmp_df = pd.read_csv(file_path)
            df_list.append(tmp_df)

        df = pd.concat(df_list, ignore_index=True)

        # read FIPS code as string with leading zero
        df["FIPS Code"] = df["FIPS Code"].astype(str).str.zfill(5)

        # filter the county and daily variables
        df = df[(df["FIPS Code"] == fips_code) & (df["Daily/Monthly"] == "Daily")]
        df.columns = df.columns.str.strip()

        group_month = df.groupby(['Month'])

        temporal_list = []
        for month, df_month in group_month:
            group_grid = df_month.groupby(['Grid Index'])

            time_series = []
            for grid, df_grid in group_grid:
                df_grid = df_grid.sort_values(by=['Day'], ascending=[True], na_position='first')

                df_grid = df_grid[df_grid.Day.isin(self.day_range)]
                df_grid = df_grid[self.column_names]
                val = torch.from_numpy(df_grid.values)
                time_series.append(val)

            temporal_list.append(torch.stack(time_series))

        x_short = torch.stack(temporal_list)
        #  m, d, g, and p represent the numbers of month, days, grids and parameters
        x_short = rearrange(x_short, 'm g d p -> m d g p')
        return x_short

    def get_long_term_val(self, fips_code, temporal_file_paths):
        """
            Return the monthly weather parameters
        :param fips_code: the unique FIPS code for on county
        :param temporal_file_paths: the file paths for CSV files
        :return:
        """

        temporal_list = []

        for file_paths in temporal_file_paths:
            df_list = []
            for file_path in file_paths:
                tmp_df = pd.read_csv(file_path)
                df_list.append(tmp_df)

            df = pd.concat(df_list, ignore_index=True)

            # read FIPS code as string with leading zero
            df["FIPS Code"] = df["FIPS Code"].astype(str).str.zfill(5)

            # filter the county and daily variables
            df = df[(df["FIPS Code"] == fips_code) & (df["Daily/Monthly"] == "Monthly")]

            df.columns = df.columns.str.strip()
            group_month = df.groupby(['Month'])

            month_list = []
            for month, df_month in group_month:
                df_month = df_month[self.column_names]
                val = torch.from_numpy(df_month.values)
                val = torch.flatten(val, start_dim=0)
                month_list.append(val)

            temporal_list.append(torch.stack(month_list))

        x_long = torch.stack(temporal_list)
        return x_long
