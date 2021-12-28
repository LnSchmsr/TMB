"""
This module implements the data sets for easy usage with pytorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import numpy as np
from numpy.fft import fft
import os
from tqdm import tqdm
import logging as log

logger = log.basicConfig(level=log.INFO)

def scale_data(series):
    """
    Scales a data series.
    """
    max_value = np.max(np.abs(series[0, :]))
    return series[0,:] / max_value


def load_compressed_data(filepath: str):
    data = np.load(filepath, allow_pickle=True)
    return data['arr_0']

def load_dataset_from_npz(filepath: str) -> np.array:
    """Loads a dataset from npz - file

    Args:
        filepath (str): Filepath of the npz-file

    Returns:
        np.array: Vibration data with the form [N,] where N is the number of __
    """
    log.info(f'Start loading')
    data = np.load(filepath, allow_pickle=True)
       
    data_list = []
    for bridge in data['arr_0']:
        f = bridge[1]
        for a in bridge[0]:
            data_list.append([a,f])
    log.info('Finished loading')
    return np.array(data_list, dtype=object)


def get_train_valid_test_loader(dataset: Dataset, valid_size: float, test_size: float, batch_size: int, num_workers: int):
    test_sampler, valid_sampler, train_sampler, test_indices = dataset.get_train_valid_test_sampler(valid_size=valid_size, test_size=test_size)

    test_loader = DataLoader(dataset, batch_size = batch_size,
                             sampler = test_sampler, num_workers = num_workers, drop_last=True)
    valid_loader = DataLoader(dataset, batch_size = batch_size,
                             sampler = valid_sampler, num_workers = num_workers, drop_last=True)
    train_loader = DataLoader(dataset, batch_size = batch_size,
                             sampler = train_sampler, num_workers = num_workers, drop_last=True)
    
    return train_loader, valid_loader, test_loader, test_indices
                             
def get_train_valid_loader(dataset: Dataset, split: float, shuffle: bool, batch_size: int, num_workers: int):
    train_sampler, valid_sampler = dataset.get_train_and_valid_sampler(split,batch_size, shuffle=shuffle)

    train_loader = DataLoader(dataset, batch_size = batch_size,
                            sampler = train_sampler, num_workers = num_workers, drop_last=True)
    valid_loader = DataLoader(dataset, batch_size = batch_size,
                            sampler = valid_sampler, num_workers = num_workers, drop_last=True)
    
    return train_loader, valid_loader

class AccelerationDataSet(Dataset):
    """
    Basis class, representing a general Dataset
    The data set has the form of:
    x: Input - Matrix [velocities, acceleration matrix] with acceleration matrix [acceleration signale, wheels/sensors]
    y: Label - Vektor: [n]
    """
    def __init__(self, filepath: str, snr = 0, sample_step_x = 1):
        """Init a base class

        Args:
            filepath (str): Path to the data in .npz - Format
            snr (int, optional): Noise Ratio. Defaults to 0.
            sample_step_x (int, optional): Sampling Rate for the data set. Defaults to 1.
        """
        #Load dataset
        self.xy = load_dataset_from_npz(filepath=filepath)     
        #Define input and output with x == input, y == output     
        self.x = self.xy[:,0]
        self.y = self.xy[:,1]
        self.n_samples = self.xy.shape[0]
        #adding noise to the input
        self.snr = snr
        self.sample_step_x = sample_step_x
        if self.snr != 0:
            self.__add_white_gaussian_noise()
    
    def __getitem__(self,index):
        x,y = torch.tensor(self.x[index][::self.sample_step_x]).float(), torch.tensor(self.y[index]).float()
        return x,y 

    def __len__(self):
        return self.n_samples
    
    def __add_white_gaussian_noise(self):
        """
        Adds white noise to the simulated un-noised acceleration signal. 
        The noise is normally distributed with a mean value of 0. 
        The amount of noise can be adjusted via the Signal to Noise ratio.

        Args:
            snr (int): Signal to noise ratio. Indicates the "strength" of the noise. Defaults to 0.
        """
        print('Start adding noise...')
        mean_noise = 0
        for i,a in enumerate(self.x):
            x_noise =[]
            for row in a.T:
                rms_a = np.sqrt(np.mean(row**2))
                std_noise = np.sqrt((rms_a**2)/(10**(self.snr/10)))
                noise = np.random.normal(mean_noise, std_noise, row.shape)
                a_noise = row + noise
                x_noise.append(a_noise)
            self.x[i] = np.vstack(tuple(x_noise)).T
        print('Finished adding noise.')
    
    def get_train_valid_test_sampler(self, valid_size: float, test_size: float):
        """Performs a train, valid and test split on the data

        Args:
            valid_size (float): Size of the validation set
            test_size (float): Size of the test set
            batch_size (int): batch size

        Returns:
            Sampler: Sampler with indices
        """
        indices = list(range(self.n_samples))
        test = int(np.floor(test_size*self.n_samples))
        valid = int(np.floor(valid_size*self.n_samples))

        np.random.shuffle(indices)
        
        test_index, valid_index, train_index = indices[:test], indices[test:test+valid], indices[test+valid:]

        return SubsetRandomSampler(test_index), SubsetRandomSampler(valid_index), SubsetRandomSampler(train_index), indices[:test]

    def get_train_and_valid_sampler(self, valid_size: float, batch_size: int, shuffle = True):
        """
        Performs a train/valid split

        Args:
            valid_size (float): Size of the validation set
            shuffle (bool, optional): Shuffle the data. Defaults to True.
            batch_size (int): Batch size of the data

        Returns:
            Sampler: Sampler with indices
        """
        if shuffle:
            indices = list(range(self.n_samples))
            np.random.shuffle(indices)
            split = int(np.floor(valid_size*self.n_samples))
            train_index, valid_index = indices[split:], indices[:split]
            return SubsetRandomSampler(train_index), SubsetRandomSampler(valid_index)
        else:
            indices = list(range(self.n_samples//batch_size))
            split = int(np.floor(valid_size*self.n_samples//batch_size))
            train_index, valid_index = indices[split:], indices[:split]
            train_index,valid_index = list(range(train_index[-1]*batch_size)), list(range(valid_index[-1]*batch_size,train_index[-1]*batch_size,1))
            return SequentialSampler(train_index), SequentialSampler(valid_index)
    
    def get_a_from_different_bridges(self,bridge_count: int, v_num: int):
        """
        Gets the acceleration matrix for the same velocities in a different measurement

        Args:
            bridge_count (int): Number of bridges in the data set
            v_num (int): Number of the velocity (e.g. 100 = 1)

        Returns:
            np.array: Acceleration matrix
        """
        splitted_a = np.split(self.x, bridge_count)
        res = [a[v_num-1] for a in splitted_a]
        return np.array(res)

class AccelerationDataSetFFT(AccelerationDataSet):
    """
    Represents the FF - transformed data set.
    
    x: Input - Matrix [velocities, acceleration matrix]
    One acceleration matrix has the form of: [signal, wheels/sensors]
    y: Frequencies [n]

    """
    def __init__(self, filepath, sampling_rate = 1000, x_scaler = None, snr=0, vec=False):
        super().__init__(filepath, snr)
        self.sampling_rate = sampling_rate
        self.freq = []
        self.x_scaler = x_scaler
        self.__fft_transformation()
        if vec == True:
            for i,v in enumerate(self.x):
                self.x[i] = np.ravel(v[:800,:])
       
        self.linear = False
        self.__transform_for_linear()
        
    def __getitem__(self, index):
        if self.linear:
            x,y = torch.tensor(self.x[:self.sampling_rate,index].T).float(),torch.tensor(self.y[index]).float()
        else:
            x,y = torch.tensor(self.x[index][:self.sampling_rate,None].T).float(), torch.tensor(self.y[index]).float()
        return x,y 
    
    def __fft_transformation(self):
        """
        Performs a FFT for each wheel of a velocity
        
        Args:
            sampling_rate (int, optional): Sampling rate. Defaults to 1000.
        """
        for i,v in enumerate(self.x):
            fft_p = []
            for row in v.T:
                f,p = self.__transform_row(row)
                self.freq.append(f)
                fft_p.append(p)
            self.x[i] = np.vstack(tuple(fft_p)).T
           

    def __transform_row(self,row: np.ndarray):
        """
        Transforms one row

        Args:
            row (np.ndarray): One row of the input
            sampling_rate (int): Sampling rate 

        Returns:
            np.array: FF-transformed row
        """
        data = row
        sr = self.sampling_rate
        X = np.abs(fft(data))
        N = len(X)
        n = np.arange(N)
        T = N/sr
        freq = n[:int(N/2+1)]/T
        P1 = 2*X[:int(N/2+1)]
        if self.x_scaler == None:
            return freq,P1
        else:
            P2 = self.x_scaler.fit_transform(P1[:,None])
            P2 = np.ravel(P2)  
            return freq,P2
    
    def __transform_for_linear(self):
        """
        Transforms input for linear layers.
        """
        res_a, res_f = [],[]
        for a,f in zip(self.x,self.y):
            for curr in a.T: 
                res_a.append(curr)
                res_f.append(f)
        self.x = np.vstack(res_a).T
        self.y = np.vstack(res_f)
        self.n_samples = len(self.y)
        self.linear = True


class AcclerationDataSetSchmutter(Dataset):
    """
    Represents the dataset for the schmutter data.
    """
    def __init__(self, filepath: str, freq_path: str, repeats_f: int, outliers = [], load_from_txt: bool = False, sampling_rate=2400, x_scaler=None):
        """Init Funktion der Basisklasse

        Args:
            filepath (str): Zum Laden der Daten
            snr (int, optional): Noise Ratio. Defaults to 0.
            sample_step_x (int, optional): Sampling Rate für den Inputdatensatz. Defaults to 1.
        """
        #Load dataset
        if load_from_txt == False:
            self.xy = load_compressed_data(filepath=filepath)    
            #Define input and output with x == input, y == output    
            # Create x Data
            x = []
            for i,(_,a) in enumerate(self.xy):
                if i not in outliers: #remove outliers
                    for a_x in a:
                        tmp = np.zeros(shape=(1,15000))
                        #Trim time series
                        a_x = np.trim_zeros(a_x, trim='fb')
                        a_x = np.reshape(a_x, (1,len(a_x)))
                        #Scale time series
                        a_x = scale_data(a_x)
                        if len(a_x) <= 15000:
                            tmp[0,:len(a_x)] = a_x.reshape((1,len(a_x)))[0,:]
                        else:
                            tmp[0,:15000] = a_x.reshape((1,len(a_x)))[0,:15000]
                        
                        x.append(tmp.T)
                        
            self.x = np.hstack(x)
        else:
            self._load_from_txt(directory_path=filepath, outliers=outliers, repeats=repeats_f)
        
        self.y = self._get_freq(freq_path, repeats=repeats_f, outliers=outliers)
        self.n_samples = self.x.shape[1]
        self.sampling_rate = sampling_rate
        self.x_scaler = x_scaler
        self.__fft_transformation()


    def _load_from_txt(self,directory_path: str, repeats:int, outliers = []):
        x=[]
        if directory_path != '':
            for i,entry in enumerate(tqdm(os.scandir(directory_path),desc='Files: ')):
                if i not in outliers:
                    for a_x in np.loadtxt(entry.path, delimiter=',').T: #15000 x 8 -> 8x15000 
                        #Trim time series
                        a_x = np.trim_zeros(a_x, trim='fb')
                        a_x = np.reshape(a_x, (1,len(a_x)))
                        #Scale time series
                        a_x = scale_data(a_x)

                        tmp = np.zeros(shape=(1,15000))
                        if len(a_x) <= 15000:
                            tmp[0,:len(a_x)] = a_x.reshape((1,len(a_x)))[0,:]
                        else:
                            tmp[0,:15000] = a_x.reshape((1,len(a_x)))[0,:15000]
                        
                        x.append(tmp.T)
            self.x = np.hstack(x)
     
        else:
            log.info('Directory was empty.')
            return 0
        

    def _get_freq(self, txt_path:str, repeats: int, outliers = []):
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            freq = [float(x) for i,x in enumerate(np.loadtxt(txt_path, delimiter=',')) if i not in outliers]
            print(freq)
            return np.repeat(freq, repeats)

    def __getitem__(self,index):
        x,y = torch.tensor(self.x[:1000,index].T).float(),torch.tensor(self.y[index]).float()
        return x,y 

    def __len__(self):
        return self.n_samples
    
    def get_train_valid_test_sampler(self, valid_size: float, test_size: float):
        indices = list(range(self.n_samples))
        test = int(np.floor(test_size*self.n_samples))
        valid = int(np.floor(valid_size*self.n_samples))

        np.random.shuffle(indices)
        
        test_index, valid_index, train_index = indices[:test], indices[test:test+valid], indices[test+valid:]

        return SubsetRandomSampler(test_index), SubsetRandomSampler(valid_index), SubsetRandomSampler(train_index), test_index
    
    def get_train_and_valid_sampler(self, valid_size: float, batch_size, shuffle):
        indices = list(range(self.n_samples))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size*self.n_samples))
        train_index, valid_index = indices[split:], indices[:split]
        return SubsetRandomSampler(train_index), SubsetRandomSampler(valid_index)

    def __fft_transformation(self):
        """
        Performs a FFT for each wheel of a velocity
        
        Args:
            sampling_rate (int, optional): Sampling rate. Defaults to 1000.
        """
        tmp = []
        for i,v in enumerate(self.x.T):
            _,p = self.__transform_row(v)
            tmp.append(p)
        
        self.x = np.vstack(tmp).T
            

    def __transform_row(self,row: np.ndarray):
        """
        Transforms one row

        Args:
            row (np.ndarray): One row of the input
            sampling_rate (int): Sampling rate 

        Returns:
            np.array: FF-transformed row
        """
        data = row
        sr = self.sampling_rate
        X = np.abs(fft(data))
        N = len(X)
        n = np.arange(N)
        T = N/sr
        freq = n[:int(N/2+1)]/T
        P1 = 2*X[:int(N/2+1)]
        if self.x_scaler == None:
            return freq,P1
        else:
            P2 = self.x_scaler.fit_transform(P1[:,None])
            P2 = np.ravel(P2)  
            return freq,P2