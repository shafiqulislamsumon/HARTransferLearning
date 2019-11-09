# encoding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
RANDOM_SEED = 42

def load_wisdm_data():
    columns = ['user','Activity','timestamp', 'Ax', 'Ay', 'Az']
    df = pd.read_csv('./dataset/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
    df = df.dropna()
    
    indexes = df[ (df['Activity'] == 'Downstairs') | (df['Activity'] == 'Upstairs') | (df['Activity'] == 'Standing') | (df['Activity'] == 'Sitting') ].index
    df.drop(indexes , inplace=True)
    
    df['Activity'] = df['Activity'].map({'Jogging': 0, 'Walking': 1})
    
    pd.options.mode.chained_assignment = None  # default='warn'
    df['Ax'] = df['Ax'] / df['Ax'].max()
    df['Ay'] = df['Ay'] / df['Ay'].max()
    df['Az'] = df['Az'] / df['Az'].max()
    # Round numbers
    df = df.round({'Ax': 4, 'Ay': 4, 'Az': 4})
    
    N_TIME_STEPS = 200
    N_FEATURES = 3
    step = 20
    segments = []
    labels = []

    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = df['Ax'].values[i: i + N_TIME_STEPS]
        ys = df['Ay'].values[i: i + N_TIME_STEPS]
        zs = df['Az'].values[i: i + N_TIME_STEPS]
        label = stats.mode(df['Activity'][i: i + N_TIME_STEPS])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    Wisdm_X_train = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    Wisdm_Y_train = np.asarray(labels, dtype = np.int32)
    Wisdm_X_train = Wisdm_X_train.reshape((-1, 3, 1, 200))
    
    print('Wisdm_X_train.shape : ', Wisdm_X_train.shape)
    print('Wisdm_Y_train shape : ', Wisdm_Y_train.shape)
    print('Wisdm_Y_train values : ', Wisdm_Y_train)
    
    return Wisdm_X_train, Wisdm_Y_train

def load_hasc_data():
    dataset_path = './dataset/HASC/'
    activity_folders = os.listdir(dataset_path)
    print(activity_folders)

    df = pd.DataFrame()
    df_list = []

    for i in range(len(activity_folders)):
        activity_folder = activity_folders[i]
        #print(activity_folder)

        person_folder_path = dataset_path + activity_folder + '/'
        person_folders = os.listdir(person_folder_path)

        for j in range(len(person_folders)):
            person_folder = person_folders[j]

            csv_file_path = dataset_path + activity_folder + '/' + person_folder + '/'
            csv_files = os.listdir(csv_file_path)

            for k in range(len(csv_files)):
                csv_file = csv_files[k]
                #print(csv_file)

                data_frame = pd.read_csv(csv_file_path + '/' + csv_file, index_col=None, header=None)
                data_frame['Activity'] = activity_folder
                df_list.append(data_frame)

    df = pd.concat(df_list, axis = 0, sort= True, ignore_index = True)
    df.columns = ['Timestamp', 'Ax', 'Ay', 'Az', 'Activity']
    
    indexes = df[ (df['Activity'] == 'Stay') | (df['Activity'] == 'Skip') | (df['Activity'] == 'Stand_Up') | (df['Activity'] == 'Sit_Down') ].index
    df.drop(indexes , inplace=True)
    df['Activity'] = df['Activity'].map({'Jogging': 0, 'Walk': 1})
    
    pd.options.mode.chained_assignment = None  # default='warn'
    df['Ax'] = df['Ax'] / df['Ax'].max()
    df['Ay'] = df['Ay'] / df['Ay'].max()
    df['Az'] = df['Az'] / df['Az'].max()
    # Round numbers
    df = df.round({'Ax': 4, 'Ay': 4, 'Az': 4})
    
    N_TIME_STEPS = 200
    N_FEATURES = 3
    step = 20
    segments = []
    labels = []

    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = df['Ax'].values[i: i + N_TIME_STEPS]
        ys = df['Ay'].values[i: i + N_TIME_STEPS]
        zs = df['Az'].values[i: i + N_TIME_STEPS]
        label = stats.mode(df['Activity'][i: i + N_TIME_STEPS])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    Hasc_X_train = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    Hasc_Y_train = np.asarray(labels, dtype = np.int32)
    Hasc_X_train = Hasc_X_train.reshape((-1, 3, 1, 200))
    
    print('Hasc_X_train.shape : ', Hasc_X_train.shape)
    print('Hasc_Y_train.shape : ', Hasc_Y_train.shape)
    print('Hasc_Y_train values : ', Hasc_Y_train)
    
    return Hasc_X_train, Hasc_Y_train

class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return self.T(sample), target

    def __len__(self):
        return len(self.samples)


def load(batch_size=64):
    source_x_train, source_y_train = load_hasc_data()
    target_x_train, target_y_train = load_wisdm_data()
    
    transform = transforms.Compose([transforms.ToTensor()])
    source_set = data_loader(source_x_train, source_y_train, transform)
    target_set = data_loader(target_x_train, target_y_train, transform)
    source_loader = DataLoader(source_set, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False)
    
    return source_loader, target_loader
