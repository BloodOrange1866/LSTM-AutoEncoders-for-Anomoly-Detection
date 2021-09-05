import pandas as pd
pd.options.mode.chained_assignment = None
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

META_LABELS = ['target', 'Date:']


def return_dataset(args: dict) -> dict:
    # Load the raw dataset
    raw_data = load_dataset(fname=args['data']['values'], skip_rows=1)
    labels = load_dataset(fname=args['data']['labels'], skip_rows=1)

    # Combine the raw and labels data, add the labels
    data_with_labels = combine_data_create_labels(
        raw_data=raw_data,
        labels=labels,
    )

    # Remove correlated features where the absolute correlation > 0.8.
    data_without_correlated_variables = remove_correlated_features(
        data=data_with_labels
    )

    # Split the dataset into train, valid, test and anomoly. Learn the imputation from the train
    # dataset and apply to the other datasets so there is no data leakage
    split_dataset = split_train_test(data=data_without_correlated_variables)

    # learn imputation by multiple chained equations
    imputed_dataset = imputed_by_multiple_imputation_by_chained_equations(
        data=split_dataset
    )

    # normalise the features
    norm_standard_data = normalise_and_standardise_data(data=imputed_dataset)

    # convert the dataset to Pytorch tensors for training models
    model_ready = convert_to_tensors(norm_standard_data)

    return model_ready


def normalise_and_standardise_data(data: dict) -> dict:
    train, test, valid, anomoly = data['train'], data['test'], data['valid'], data['anomoly']

    norm = MinMaxScaler().fit(train)
    train.loc[:,:] = norm.transform(train)
    test.loc[:,:] = norm.transform(test)
    valid.loc[:,:] = norm.transform(valid)

    norm = MinMaxScaler().fit(anomoly)
    anomoly.loc[:,:] = norm.transform(anomoly)

    scale = StandardScaler().fit(train)
    train.loc[:,:] = scale.transform(train)
    test.loc[:,:] = scale.transform(test)
    valid.loc[:,:] = scale.transform(valid)

    scale = StandardScaler().fit(anomoly)
    anomoly.loc[:,:] = scale.transform(anomoly)

    return {
        'train': train,
        'test': test,
        'valid': valid,
        'anomoly': anomoly
    }

def convert_to_tensors(data: dict) -> dict:

    def convert_using_pytorch(data: pd.DataFrame):
        sequences = data.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features

    anomoly, seq_len, n_features = convert_using_pytorch(data['anomoly'])
    train, _, _ = convert_using_pytorch(data['train'])
    valid, _, _ = convert_using_pytorch(data['valid'])
    test, _, _ = convert_using_pytorch(data['test'])

    return {
        'anomoly': anomoly,
        'train': train,
        'test': test,
        'valid': valid,
        'seq_len': seq_len,
        'n_features': n_features,
    }

def split_train_test(data: pd.DataFrame) -> dict:
    anomoly_dataset = data[data['target'] == 1]
    normal_dataset = data[data['target'] == 0]

    train, valid = train_test_split(
        normal_dataset,
        test_size=0.15,
        random_state=42
    )

    valid, test = train_test_split(
        valid,
        test_size=0.33,
        random_state=42
    )

    return {
        'anomoly': anomoly_dataset,
        'train': train,
        'test': test,
        'valid': valid
    }


def imputed_by_multiple_imputation_by_chained_equations(data: dict) -> dict:
    anomoly, train, test, valid = data['anomoly'], data['train'], data['test'], data['valid']

    train = remove_meta_cols(train)
    test = remove_meta_cols(test)
    anomoly = remove_meta_cols(anomoly)
    valid = remove_meta_cols(valid)

    imp = IterativeImputer(max_iter=100, random_state=0)
    model = imp.fit(train)

    train.loc[:,:] = model.transform(train)
    test.loc[:,:] = model.transform(test)
    anomoly.loc[:,:] = model.transform(anomoly)
    valid.loc[:,:] = model.transform(valid)

    return {
        'anomoly': anomoly,
        'train': train,
        'test': test,
        'valid': valid
    }



def remove_correlated_features(data: pd.DataFrame) -> pd.DataFrame:
    feats = data[[c for c in data.columns if c not in META_LABELS]]
    feats = feats.astype(float)
    correlation = feats.corr()

    feats_to_remove = set()
    for col in correlation.columns:
        for row in correlation.index.values:
            if col != row:
                corr = correlation.at[row, col]
                if abs(corr) > 0.8:
                    feats_to_remove.add(col)

    feats = feats[[f for f in feats.columns if f not in list(feats_to_remove)]]
    feats[META_LABELS] = data[META_LABELS]
    return feats



def load_dataset(fname: str, skip_rows: int) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(os.path.dirname(os.getcwd()), 'data', fname), header=None, skiprows=skip_rows
    )


def combine_data_create_labels(raw_data: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    clean_labels = [feat.split(' ')[0] for feat in labels[0].tolist()] # clean labels
    clean_labels.append('target')
    raw_data = raw_data.replace('?', np.nan)
    raw_data.columns = clean_labels
    return raw_data


def remove_meta_cols(data: pd.DataFrame) -> pd.DataFrame:
    return data[[c for c in data.columns if c not in META_LABELS]]