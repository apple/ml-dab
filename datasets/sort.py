#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.utils.data
import numpy as np


def generate_samples(num_samples, seq_len, max_digit):
    """ Helper to generate sampels between 0 and max_digit

    :param num_samples: the total number of samples to generate
    :param seq_len: length of each sequence
    :param max_digit: the upper bound in the uniform distribution
    :returns: [B, seq_len]
    :rtype: torch.Tensor, torch.Tensor

    """
    data = np.random.uniform(0, max_digit, size=[num_samples, seq_len])
    labels = np.argsort(data, axis=-1)
    data = data.reshape(num_samples, seq_len, 1)
    labels = labels.reshape(num_samples, seq_len*1)
    print('[debug] labels = ', labels.shape, " | data = ", data.shape)
    return [data.astype(np.float32), labels]


class SortDataset(torch.utils.data.Dataset):
    def __init__(self, upper_bound_unif, sequence_length, split='train',
                 transform=None, target_transform=None, **kwargs):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.max_digit = upper_bound_unif       # max sorting range U ~ [0, max_digit]
        self.sequence_length = sequence_length  # set the sequence length if it isn't specified

        # set the number of samples to 2 million by default
        train_samples = kwargs.get('num_samples', 2000000)
        self.num_samples = train_samples if split == 'train' else int(train_samples*0.2)

        # load the sort dataset and labels
        self.data, self.labels = generate_samples(self.num_samples,
                                                  self.sequence_length,
                                                  self.max_digit)
        print("[{}] {} samples\n".format(split, len(self.labels)))

    def __getitem__(self, index):
        """ Returns a single element based on the index.
            Extended by pytorch to a queue based loader.

        :param index: the single sample id
        :returns: an unsorted vector and the correct sorted class target.
        :rtype: torch.Tensor, torch.Tensor

        """
        target = self.labels[index]
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # sanity conversions in case the data has not yet been
        # converted to a torch.Tensor
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target)

        return data, target

    def __len__(self):
        return len(self.labels)


class SortLoader(object):
    def __init__(self, batch_size, upper_bound_unif, sequence_length, transform=None, target_transform=None,  **kwargs):
        """ A container class that houses a train and test loader for the sort problem.

        :param batch_size: the minibatch size
        :param upper_bound_unif: the upper bound in U(0, upper_bound_unif)
        :param sequence_length: how many samples in input?
        :param transform: torchvision transforms if needed
        :param target_transform: torchvision target label transforme
        :returns: SortLoader object with .train_loader and .test_loader to iterate corresponding datasets
        :rtype: object

        """
        # build the datasets that implement __getitem__
        train_dataset = SortDataset(upper_bound_unif, sequence_length, split='train',
                                    transfor=transform, target_transform=target_transform, **kwargs)
        test_dataset = SortDataset(upper_bound_unif, sequence_length, split='test',
                                   transfor=transform, target_transform=target_transform, **kwargs)

        # build the dataloaders that wrap the dataset
        loader_args = {'num_workers': 4, 'pin_memory': True, 'batch_size': batch_size, 'drop_last': True}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **loader_args
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, **loader_args
        )
