import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def load_data(folder_path, folder_name):
    return pd.read_csv(os.path.join(folder_path, folder_name))

def clean():
    data = load_data(os.environ.get('FOLDER_PATH'), os.environ.get('DATASET_FILE'))
    type_mapping = {
        'N': 'Normal',
        'SVEB': 'Abnormal',
        'VEB': 'Abnormal',
        'F': 'Abnormal',
        'Q': 'Abnormal'
    }
    data = data.drop(columns=['record'])
    data['type'] = data['type'].map(type_mapping)
    return data

def preprocess():
    data = clean()
    X = data.drop('type', axis=1)
    y = data['type']
    binary_type_mapping = {'Normal': 0, 'Abnormal': 1}
    y = y.map(binary_type_mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    subset_size = 50000
    X_train_subsample = X_train_resampled[:subset_size]
    y_train_subsample = y_train_resampled[:subset_size]

    return X_train_subsample, y_train_subsample, X_test, y_test, scaler
