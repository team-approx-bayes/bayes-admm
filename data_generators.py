# Data generators
import torch
import numpy as np
from sklearn.datasets import make_blobs#, make_moons
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import pickle
import gzip


# 2D binary classification with Gaussian blobs
# setting==1: the original task from FROMP paper (5 clients with Gaussian blobs)
# setting==2: 2 clients, one much more local than the other
# setting==3: 2 clients, in different but overlapping regions of input-space
# num_clients: total number of data splits to consider
# num_samples: number of samples from each Gaussian blob
class ToyDataGenerator():
    def __init__(self, setting=1, num_clients=5, num_samples=2000, polynomial_degree=1, seed=0, multiclass=False,
                 path=None):

        self.state = {}  # Everything required for recreating this data_generator (can save this in a file and load it)
        self.without_trainloaders = True  # If True, do not use trainloaders

        # Load self.state from path
        if path is not None:
            self.state = torch.load(path)['data_generator']

        # Not loading state from path
        else:
            self.state['seed'] = seed
            self.state['setting'] = setting
            self.state['num_clients'] = num_clients
            self.state['num_samples'] = num_samples
            self.state['polynomial_degree'] = polynomial_degree
            self.state['multiclass'] = multiclass

            # Centers and std devs of the 10 Gaussian blobs
            # There are up to 5 binary classification tasks here
            if setting == 1:
                self.state['gaussian_centers'] = [[[0.0, 0.2], [0.45, 0]],
                                                  [[0.6, 0.9], [0.7, 0.45]],
                                                  [[1.3, 0.4], [1.0, 0.1]],
                                                  [[1.6, -0.1], [1.7, -0.4]],
                                                  [[2.0, 0.3], [2.3, 0.1]]]

                self.state['gaussian_std'] = [[[0.08, 0.22], [0.08, 0.16]],
                                              [[0.24, 0.08], [0.16, 0.08]],
                                              [[0.04, 0.20], [0.06, 0.16]],
                                              [[0.16, 0.05], [0.24, 0.05]],
                                              [[0.05, 0.16], [0.05, 0.22]]]

            if setting == 2:
                # Two clients. First is spread out, second is much more concentrated.
                # First has a slightly different position for decision boundary than the second, and training on the
                # second should fix this.
                self.state['gaussian_centers'] = [[[0.0, 0], [1, 0]],
                                                  [[0.65, 0], [0.85, 0]]]

                self.state['gaussian_std'] = [[[0.25, 0.25], [0.25, 0.25]],
                                              [[0.05, 0.1], [0.05, 0.1]]]

            if setting == 3:
                # Two clients. They are in different but overlapping regions of input-space
                self.state['gaussian_centers'] = [[[0.0, 0], [1, 0]],
                                                  [[0.5, 0], [1.5, 0]]]

                self.state['gaussian_std'] = [[[0.2, 0.25], [0.2, 0.25]],
                                              [[0.2, 0.25], [0.2, 0.25]]]

        if self.state['num_clients'] > 5:
            raise Exception("Current ToyDataGenerator only supports up to 5 tasks.")
        if (self.state['setting'] == 2 or self.state['setting'] == 3) and self.state['num_clients'] > 2:
            raise Exception("Current ToyDataGenerator with setting==2 or ==3 only supports up to 2 tasks.")

        # Create data
        self.create_data()

    # Create datasets
    def create_data(self):

        np.random.seed(self.state['seed'])
        self.offset = self.state['num_clients']  # Offset when loading data in next_task()

        self.num_points_per_client = []  # Number of datapoints in each client

        # Generate the Gaussian blobs
        self.inputs_plot = []
        self.labels = []
        for data_split_ind in range(self.state['num_clients']):
            inputs_interm, labels_interm = make_blobs(self.state['num_samples']*2,
                                                      centers=self.state['gaussian_centers'][data_split_ind],
                                                      cluster_std=self.state['gaussian_std'][data_split_ind],
                                                      shuffle=False)
            self.inputs_plot.append(inputs_interm.astype('float32'))

            # Multiclass labels vs binary classification
            if self.state['multiclass']:
                self.labels.append(labels_interm + 2*data_split_ind)
            else:
                self.labels.append(labels_interm)

            # Number of datapoints in client
            self.num_points_per_client.append(len(labels_interm))

        self.inputs_plot = np.array(self.inputs_plot)
        self.labels = np.array(self.labels)

        # Find the range covered in 2D space
        self.x_axis_min, self.x_axis_max = self.inputs_plot[:, :, 0].min() - 0.2, self.inputs_plot[:, :, 0].max() + 0.2
        self.y_axis_min, self.y_axis_max = self.inputs_plot[:, :, 1].min() - 0.2, self.inputs_plot[:, :, 1].max() + 0.2

        # Make a meshgrid covering the entire 2D range (for plotting and for testing)
        h = 0.01
        x_axis_mesh, y_axis_mesh = np.meshgrid(np.arange(self.x_axis_min, self.x_axis_max, h),
                                               np.arange(self.y_axis_min, self.y_axis_max, h))
        x_axis_mesh = x_axis_mesh.astype('float32')
        y_axis_mesh = y_axis_mesh.astype('float32')
        self.test_shape = x_axis_mesh.shape
        inputs_test = np.c_[x_axis_mesh.ravel(), y_axis_mesh.ravel()]
        self.inputs_test = torch.from_numpy(inputs_test)
        self.labels_test = torch.zeros((len(self.inputs_test)), dtype=self.inputs_test.dtype)

        # Polynomial features
        if self.state['polynomial_degree'] is not None:
            self.poly = PolynomialFeatures(self.state['polynomial_degree'])
            self.inputs = self.poly.fit_transform(self.inputs_plot.reshape(
                -1,self.inputs_plot.shape[-1])).reshape(self.inputs_plot.shape[0],self.inputs_plot.shape[1],-1)
            self.inputs_test = torch.from_numpy(
                PolynomialFeatures(self.state['polynomial_degree']).fit_transform(inputs_test))
        else:
            self.inputs = self.inputs_plot

        # List of classes seen at each task
        if self.state['multiclass']:
            self.num_outputs = np.max(self.labels) + 1  # TODO: test this is correct
            self.num_parameters = self.inputs.shape[2]
        else:
            self.num_outputs = 2
            self.num_parameters = self.inputs.shape[2]


    # Return trainloader and testloader for a specific client_ind
    def data_split(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            if self.state['multiclass']:
                # Inputs of training data
                inputs_train_class_0 = self.inputs[client_ind, self.labels[client_ind] == 0 + 2 * client_ind]
                inputs_train_class_1 = self.inputs[client_ind, self.labels[client_ind] == 1 + 2 * client_ind]
                inputs_train = np.concatenate([inputs_train_class_0, inputs_train_class_1], axis=0)
                inputs_train = torch.from_numpy(inputs_train)

                # Labels of training data
                labels_train_class_0 = np.zeros_like(
                    self.labels[client_ind, self.labels[client_ind] == 0 + 2 * client_ind]) + 2 * client_ind
                labels_train_class_1 = np.ones_like(
                    self.labels[client_ind, self.labels[client_ind] == 1 + 2 * client_ind]) + 2 * client_ind
                labels_train = np.concatenate([labels_train_class_0, labels_train_class_1], axis=0)
                labels_train = labels_train.astype('int64')
                labels_train = torch.from_numpy(labels_train)
            else:
                # Inputs of training data
                inputs_train_class_0 = self.inputs[client_ind, self.labels[client_ind] == 0]
                inputs_train_class_1 = self.inputs[client_ind, self.labels[client_ind] == 1]
                inputs_train = np.concatenate([inputs_train_class_0, inputs_train_class_1], axis=0)
                inputs_train = torch.from_numpy(inputs_train)

                # Labels of training data
                labels_train_class_0 = np.zeros_like(self.labels[client_ind, self.labels[client_ind] == 0])
                labels_train_class_1 = np.ones_like(self.labels[client_ind, self.labels[client_ind] == 1])
                labels_train = np.concatenate([labels_train_class_0, labels_train_class_1], axis=0)
                labels_train = labels_train.astype('int64')
                labels_train = torch.from_numpy(labels_train)

            return (inputs_train, labels_train), (self.inputs_test, self.labels_test)

    # Return trainloader and testloader for a specific client_ind, for plotting (ie no polynomial features)
    def data_split_plot(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            if self.state['multiclass']:
                # Inputs of training data
                inputs_train_class_0 = self.inputs_plot[client_ind, self.labels[client_ind] == 0 + 2 * client_ind]
                inputs_train_class_1 = self.inputs_plot[client_ind, self.labels[client_ind] == 1 + 2 * client_ind]
                inputs_train = np.concatenate([inputs_train_class_0, inputs_train_class_1], axis=0)
                inputs_train = torch.from_numpy(inputs_train)

                # Labels of training data
                labels_train_class_0 = np.zeros_like(
                    self.labels[client_ind, self.labels[client_ind] == 0 + 2 * client_ind]) + 2 * client_ind
                labels_train_class_1 = np.ones_like(
                    self.labels[client_ind, self.labels[client_ind] == 1 + 2 * client_ind]) + 2 * client_ind
                labels_train = np.concatenate([labels_train_class_0, labels_train_class_1], axis=0)
                labels_train = labels_train.astype('int64')
                labels_train = torch.from_numpy(labels_train)
            else:
                # Inputs of training data
                inputs_train_class_0 = self.inputs_plot[client_ind, self.labels[client_ind] == 0]
                inputs_train_class_1 = self.inputs_plot[client_ind, self.labels[client_ind] == 1]
                inputs_train = np.concatenate([inputs_train_class_0, inputs_train_class_1], axis=0)
                inputs_train = torch.from_numpy(inputs_train)

                # Labels of training data
                labels_train_class_0 = np.zeros_like(self.labels[client_ind, self.labels[client_ind] == 0])
                labels_train_class_1 = np.ones_like(self.labels[client_ind, self.labels[client_ind] == 1])
                labels_train = np.concatenate([labels_train_class_0, labels_train_class_1], axis=0)
                labels_train = labels_train.astype('int64')
                labels_train = torch.from_numpy(labels_train)

            return (inputs_train, labels_train), (self.inputs_test, self.labels_test)

    # Return all data in dataloader
    def full_data(self):

        inputs_train = []
        labels_train = []
        for client_ind in range(self.state['num_clients']):
            if client_ind == 0:
                inputs_train = self.inputs[client_ind]
                labels_train = self.labels[client_ind]
            else:
                inputs_train = np.concatenate((inputs_train, self.inputs[client_ind]))
                labels_train = np.concatenate((labels_train, self.labels[client_ind]))

        inputs_train = torch.from_numpy(inputs_train)
        labels_train = labels_train.astype('int64')
        labels_train = torch.from_numpy(labels_train)

        return (inputs_train, labels_train), (self.inputs_test, self.labels_test)

    # Return all data for plotting (ie no polynomial features)
    def full_data_plot(self):

        inputs_train = []
        labels_train = []
        for client_ind in range(self.state['num_clients']):
            if client_ind == 0:
                inputs_train = self.inputs_plot[client_ind]
                labels_train = self.labels[client_ind]
            else:
                inputs_train = np.concatenate((inputs_train, self.inputs_plot[client_ind]))
                labels_train = np.concatenate((labels_train, self.labels[client_ind]))

        inputs_train = torch.from_numpy(inputs_train)
        labels_train = labels_train.astype('int64')
        labels_train = torch.from_numpy(labels_train)

        return (inputs_train, labels_train)


# FLamby Heart Disease dataset
# num_clients: total number of clients to consider
class FLambyHeartDataGenerator():
    def __init__(self, num_clients=4, seed=0, path=None):

        self.state = {}  # Everything required for recreating this data_generator (can save this in a file and load it)
        self.without_trainloaders = True  # If True, do not use trainloaders

        # Load self.state from path
        if path is not None:
            self.state = torch.load(path)['data_generator']

        # Generate new data
        else:
            self.state['seed'] = seed
            self.state['num_clients'] = num_clients

        if self.state['num_clients'] > 4:
            raise Exception("FLamby Heart Disease only supports up to 4 clients.")

        np.random.seed(self.state['seed'])
        from data_generators_flamby import FedHeartDisease

        # Load the centers using FLamby code, and then format according to my code
        self.inputs = []
        self.labels = []
        for client_ind in range(self.state['num_clients']):
            center_flamby = FedHeartDisease(center=client_ind, train=True)
            self.inputs.append(torch.stack(center_flamby.features))
            self.labels.append(torch.stack(center_flamby.labels).reshape(-1).long())

        # Test data
        center_test = FedHeartDisease(train=False, pooled=True)
        self.inputs_test = torch.stack(center_test.features)
        self.labels_test = torch.stack(center_test.labels).reshape(-1).long()

        # List of classes seen at each task
        self.classes = [[0, 1], [0, 1], [0, 1], [0, 1]]
        self.number_classes = [2, 2, 2, 2]
        self.num_outputs = 2
        self.num_parameters = self.inputs[0].shape[1]
        self.classes_seen = self.classes
        self.client_sizes = [199, 172, 30, 85]
        self.num_points_per_client = self.client_sizes


    # Return trainloader and testloader for a specific split_ind
    def data_split(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            # Inputs of training data
            inputs_train = self.inputs[client_ind]

            # Labels of training data
            labels_train = self.labels[client_ind]

            return (inputs_train, labels_train), (self.inputs_test, self.labels_test)


    # Return all data in dataloader
    def full_data(self, task_id=None, inputs_only=False):
        if task_id is None:
            task_id = self.state['num_clients']-1

        inputs_train = []
        labels_train = []
        for split_ind in range(task_id+1):
            if split_ind == 0:
                inputs_train = self.inputs[split_ind]
                labels_train = self.labels[split_ind]
            else:
                inputs_train = torch.concatenate((inputs_train, self.inputs[split_ind]))
                labels_train = torch.concatenate((labels_train, self.labels[split_ind]))

        if inputs_only:
            return inputs_train
        else:
            return (inputs_train, labels_train), (self.inputs_test, self.labels_test)



# Federated UCI credit classification: K clients
# num_clients: total number of data splits (=K) to consider
# data_split: 'iid' or 'non_iid' (homog vs heterog split of data)
class FederatedUCICreditDataGenerator():
    def __init__(self, num_clients=10, data_split='non_iid', seed=0, path=None):

        self.state = {}  # Everything required for recreating this data_generator (can save this in a file and load it)
        self.without_trainloaders = True  # If True, do not use trainloaders

        # Load self.state from path
        if path is not None:
            self.state = torch.load(path)['data_generator']
            num_clients = self.state['num_clients']

        # Generate new data
        else:
            self.state['seed'] = seed
            self.state['num_clients'] = num_clients

        # # Use ucimlrepo to get data
        # from ucimlrepo import fetch_ucirepo
        # credit_approval = fetch_ucirepo(id=27)
        # X = np.array(credit_approval.data.features)
        # y_interm = np.array(credit_approval.data.targets).reshape(-1)
        # X_to_onehot_encode_inds = [2,3,5,6,8,9,10,11,14]
        # X_not_onehot_encode_inds = [0,1,4,7,12,13]

        # Use stored data
        import pandas as pd
        credit_approval = pd.read_csv("data/credit+approval/crx.data")
        features_df = credit_approval.iloc[:,:-1]
        features_df = features_df.replace({'?': np.nan})
        X = np.array(features_df)
        y_interm = np.array(credit_approval.iloc[:,-1:]).reshape(-1)
        X_to_onehot_encode_inds = [0,3,4,5,6,8,9,11,12]
        X_not_onehot_encode_inds = [1,2,7,10,13,14]

        # Delete inputs which have any missing values
        nan_inds = np.where(features_df.isna())[0]
        X = np.delete(X, nan_inds, axis=0)
        y_interm = np.delete(y_interm, nan_inds)

        # One-hot encode the relevant (categorical) features
        from sklearn.preprocessing import OneHotEncoder
        X_to_onehot_encode = X[:,X_to_onehot_encode_inds]
        enc = OneHotEncoder(drop='if_binary').fit(X_to_onehot_encode)
        X_onehot_encoded = enc.transform(X_to_onehot_encode).toarray()
        X_not_onehot_encoded = X[:, X_not_onehot_encode_inds]
        X = np.concatenate((X_onehot_encoded.astype(np.float32), X_not_onehot_encoded.astype(np.float32)), axis=1)

        # Convert y_interm into 0-1 labels
        y = np.zeros_like(y_interm)
        for ind in range(len(y_interm)):
            if y_interm[ind] == '+':
                y[ind] = 1
        y = y.astype(int)

        # Train test split (equal proportion of positive/negative examples in train and test split)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.train_label, self.test_label = train_test_split(X.astype(np.float32), y.astype(np.int64),
                                                                                        test_size=0.2, random_state=1, stratify=y.astype(np.int64))

        self.binary_classification = False  # If False, this is multinomial classification

        # Binary classification on every client
        self.sets = [[0, 1] for _ in range(num_clients)]

        self.max_iter = len(self.sets)

        self.number_classes = []  # The number of classes observed so far (at task i)
        self.classes_seen_total = []  # List of all classes observed so far up to and including task i
        self.out_dim = 0  # Total number of unique classes
        self.class_list = []  # List of unique classes being considered, in the order they appear
        for task_id in range(self.max_iter):
            for class_index in range(len(self.sets[task_id])):
                if self.sets[task_id][class_index] not in self.class_list:
                    # Convert from MNIST digit numbers to class index number by using self.class_list.index(), which is done in self.classes
                    self.class_list.append(self.sets[task_id][class_index])
                    self.out_dim = self.out_dim + 1
            self.number_classes.append(self.out_dim)
            self.classes_seen_total.append([i for i in range(self.out_dim)])

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes_seen_per_task = []
        for task_id in range(self.max_iter):
            class_idx = []
            for i in range(len(self.sets[task_id])):
                class_idx.append(self.class_list.index(self.sets[task_id][i]))
            self.classes_seen_per_task.append(class_idx)

        self.dimensions = self.X_train.shape[1]
        self.num_parameters = self.dimensions
        self.classes = self.classes_seen_total  # Predict over all classes seen so far
        self.num_outputs = 2

        # This holds the indices of the datapoints for each client, split by class
        self.class_client_splits = {}
        self.class_client_splits_number = {}  # This holds the number of datatpoints from each class to store in each client
        for client_ind in range(num_clients):
            self.class_client_splits[client_ind] = {}
            self.class_client_splits_number[client_ind] = {}
            for class_ind in range(2):
                self.class_client_splits[client_ind][class_ind] = []
                self.class_client_splits_number[client_ind][class_ind] = []

        # Assign how much data we want from each class in each client (fill self.class_client_splits_number)
        # If 'iid', assign equal number of datapoints from each class to each client
        # If 'non_iid', do split like in PVI paper
        # data_split = 'non_iid'# {'iid', 'non_iid'}

        if data_split == 'iid':
            if self.state['num_clients'] == 2:
                class_0_client = [142,142]
                class_1_client = [118,118]
            else:
                class_0_client = [28,28,28,28,28,28,28,28,28,28]
                class_1_client = [23,23,23,24,24,24,24,24,24,24]
        elif data_split == 'non_iid':
            class_0_client = [34,34,34,34,34,22,22,22,22,22]
            class_1_client = [2, 2, 2, 2, 2, 45,45,45,45,45]


        # Assign the data to each client, split by how much data we want from each class
        for class_ind in range(2):
            indices_in_class = np.where(self.train_label == class_ind)[0]
            # num_in_class = len(indices_in_class)
            np.random.shuffle(indices_in_class)  # Shuffle dataset

            num_assigned_so_far = 0
            for client_ind in range(num_clients):
                if class_ind == 0:
                    num_assign_now = class_0_client[client_ind]
                else:
                    num_assign_now = class_1_client[client_ind]
                self.class_client_splits[client_ind][class_ind] = indices_in_class[num_assigned_so_far:num_assigned_so_far + num_assign_now]
                num_assigned_so_far += num_assign_now
                print(client_ind, class_ind, num_assign_now)

        # This holds all the indices for each client
        self.client_indices_list = {}
        self.client_sizes = []
        for client_ind in range(num_clients):
            self.client_indices_list[client_ind] = np.array([], dtype=np.int64)
            for class_ind in range(2):
                self.client_indices_list[client_ind] = np.concatenate((self.client_indices_list[client_ind],
                                                                       self.class_client_splits[client_ind][class_ind]))

            self.client_sizes.append(len(self.client_indices_list[client_ind]))

        self.num_points_per_client = self.client_sizes


    # Return trainloader and testloader for a specific split_ind
    def data_split(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            next_x_train = self.X_train[self.client_indices_list[client_ind]]
            next_y_train = self.train_label[self.client_indices_list[client_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            return (inputs_train, labels_train), (inputs_test, labels_test)

    # Do not return TensorDatasets, rather just the tensors, for training data (useful for removing specific points)
    def data_split_arrays_only(self, split_ind):
        if split_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            next_x_train = self.X_train[self.client_indices_list[split_ind]]
            next_y_train = self.train_label[self.client_indices_list[split_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            return (inputs_train, labels_train), TensorDataset(inputs_test, labels_test)


    # Return all data in dataloader
    def full_data(self, task_id=None):

        if task_id is None:
            task_id = self.state['num_clients']-1

        next_x_train = None
        next_y_train = None
        next_x_test = None
        next_y_test = None

        for split_ind in range(task_id+1):
            # next_x_train = self.X_train[self.client_indices_list[split_ind]]
            # next_y_train = self.train_label[self.client_indices_list[split_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            if next_x_train is None:
                next_x_train = self.X_train[self.client_indices_list[split_ind]]
            else:
                next_x_train = np.vstack((next_x_train, self.X_train[self.client_indices_list[split_ind]]))

            if next_y_train is None:
                next_y_train = self.train_label[self.client_indices_list[split_ind]]
            else:
                next_y_train = np.hstack((next_y_train, self.train_label[self.client_indices_list[split_ind]]))

        inputs_train = torch.from_numpy(next_x_train)
        labels_train = torch.from_numpy(next_y_train)
        inputs_test = torch.from_numpy(next_x_test)
        labels_test = torch.from_numpy(next_y_test)

        return (inputs_train, labels_train), (inputs_test, labels_test)


# Federated MNIST classification: K clients
# num_clients: total number of data splits (=K) to consider
# dataset_proportion: if < 1, how much of dataset to include in setup (eg 0.1)
# fashionmnist: if FashionMNIST dataset (and not MNIST)
# data_split: 'iid' or 'non_iid' (homog vs heterog split of data)
# heterog_alpha_1: If 'non_iid', Dir(alpha_1) is num points per client (high == equal split)
# heterog_alpha_2: If 'non_iid', Dir(alpha_2) is num points per class within each client (high == equal split)
class FederatedMNISTDataGenerator():
    def __init__(self, num_clients=10, seed=0, dataset_proportion=1., fashionmnist=False, data_split='non_iid',
                 heterog_alpha_1=1, heterog_alpha_2=0.5, path=None):

        self.state = {}  # Everything required for recreating this data_generator (can save this in a file and load it)

        # Load self.state from path
        if path is not None:
            self.state = torch.load(path)['data_generator']
            num_clients = self.state['num_clients']
            dataset_proportion = self.state['dataset_proportion']
            fashionmnist = self.state['fashionmnist']

        # Generate new data
        else:
            self.state['seed'] = seed
            self.state['num_clients'] = num_clients
            self.state['dataset_proportion'] = dataset_proportion
            self.state['fashionmnist'] = fashionmnist

        if fashionmnist:
            train_set = datasets.FashionMNIST('./data', download=True, train=True)
            test_set = datasets.FashionMNIST('./data', download=True, train=False)
            self.X_train = train_set.data.numpy().astype(np.float32)
            self.X_train = self.X_train.reshape(self.X_train.shape[0],-1)
            self.train_label = train_set.targets.numpy()
            self.X_test = test_set.data.numpy().astype(np.float32)
            self.X_test = self.X_test.reshape(self.X_test.shape[0],-1)
            self.test_label = test_set.targets.numpy()
        else:
            # train_set = datasets.MNIST('data', download=False, train=True)
            # test_set = datasets.MNIST('data', download=False, train=False)
            # self.X_train = train_set.data.numpy().astype(np.float32)
            # self.X_train = self.X_train.reshape(self.X_train.shape[0],-1)
            # self.train_label = train_set.targets.numpy()
            # self.X_test = test_set.data.numpy().astype(np.float32)
            # self.X_test = self.X_test.reshape(self.X_test.shape[0],-1)
            # self.test_label = test_set.targets.numpy()
            f = gzip.open('data/mnist.pkl.gz', 'rb')
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            f.close()
            self.X_train = np.vstack((train_set[0], valid_set[0]))
            self.X_test = test_set[0]
            self.train_label = np.hstack((train_set[1], valid_set[1]))
            self.test_label = test_set[1]

        self.binary_classification = False  # If False, this is multinomial classification
        self.without_trainloaders = True  # If True, do not use trainloaders

        self.dimensions = self.X_train.shape[1]
        self.num_parameters = self.dimensions
        self.num_outputs = 10

        # This holds the indices of the datapoints for each client, split by class
        self.class_client_splits = {}
        self.class_client_splits_number = {}  # This holds the number of datatpoints from each class to store in each client
        for client_ind in range(num_clients):
            self.class_client_splits[client_ind] = {}
            self.class_client_splits_number[client_ind] = {}
            for class_ind in range(10):
                self.class_client_splits[client_ind][class_ind] = []
                self.class_client_splits_number[client_ind][class_ind] = []

        # Assign how much data we want from each class in each client (fill self.class_client_splits_number)
        if data_split == 'iid':
            alpha_1 = 1000000000000000
            alpha_2 = 1000000000000000
        elif data_split == 'non_iid':
            alpha_1 = heterog_alpha_1  # Uneven distribution of number of points per client
            alpha_2 = heterog_alpha_2  # Uneven distribution of classes within each client. Can try alpha_2<1 too (eg 0.5 or 0.3 or 0.1)
        elif data_split == 'non_iid_1_05':
            alpha_1 = 1  # Uneven distribution of number of points per client
            alpha_2 = 0.5  # Uneven distribution of classes within each client. Can try alpha_2<1 too (eg 0.5 or 0.3 or 0.1)
        else:
            alpha_1 = 1000000000000000  # Dir(alpha_1) is num points per client (high == equal split)
            alpha_2 = 1000000000000000  # Dir(alpha_2) is num points per class within each client (high == equal split)
            print("Did not set data_split variable correctly in the data generator. Assuming iid split / distribution.")

        client_splits = np.random.dirichlet([alpha_1 for _ in range(num_clients)])
        client_class_splits = np.random.dirichlet([alpha_2 for _ in range(10)], (num_clients))
        normaliser_when_assigning_number = np.zeros((10))
        for client_ind in range(num_clients):
            self.class_client_splits_number[client_ind] = client_splits[client_ind] * client_class_splits[client_ind]
            for class_ind in range(10):
                normaliser_when_assigning_number[class_ind] += self.class_client_splits_number[client_ind][class_ind]

        # Assign the data to each client, split by how much data we want from each class
        for class_ind in range(10):
            indices_in_class = np.where(self.train_label == class_ind)[0]
            num_in_class = len(indices_in_class) * dataset_proportion  # Only use a proportion of the dataset
            np.random.shuffle(indices_in_class)  # Shuffle dataset

            num_assigned_so_far = 0
            for client_ind in range(num_clients):
                num_assign_now = np.round(num_in_class * self.class_client_splits_number[client_ind][class_ind] / normaliser_when_assigning_number[class_ind]).astype(int)
                if num_assign_now == 1:
                    num_assign_now = 2
                self.class_client_splits[client_ind][class_ind] = indices_in_class[num_assigned_so_far:num_assigned_so_far + num_assign_now]
                num_assigned_so_far += num_assign_now
                print(client_ind, class_ind, num_assign_now)

        # This holds all the indices for each client
        self.client_indices_list = {}
        self.num_points_per_client = []
        for client_ind in range(num_clients):
            self.client_indices_list[client_ind] = np.array([], dtype=np.int64)
            for class_ind in range(10):
                self.client_indices_list[client_ind] = np.concatenate((self.client_indices_list[client_ind],
                                                                       self.class_client_splits[client_ind][class_ind]))

            self.num_points_per_client.append(len(self.client_indices_list[client_ind]))


    # Return trainloader and testloader for a specific split_ind
    def data_split(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            next_x_train = self.X_train[self.client_indices_list[client_ind]]
            next_y_train = self.train_label[self.client_indices_list[client_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            return (inputs_train, labels_train), (inputs_test, labels_test)

    # Do not return TensorDatasets, rather just the tensors, for training data (useful for removing specific points)
    def data_split_arrays_only(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            next_x_train = self.X_train[self.client_indices_list[client_ind]]
            next_y_train = self.train_label[self.client_indices_list[client_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            return (inputs_train, labels_train), TensorDataset(inputs_test, labels_test)


    # Return all data in dataloader
    def full_data(self, task_id=None, inputs_only=False):

        if task_id is None:
            task_id = self.state['num_clients']-1

        next_x_train = None
        next_y_train = None
        next_x_test = None
        next_y_test = None

        for split_ind in range(task_id+1):
            # next_x_train = self.X_train[self.client_indices_list[split_ind]]
            # next_y_train = self.train_label[self.client_indices_list[split_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            if next_x_train is None:
                next_x_train = self.X_train[self.client_indices_list[split_ind]]
            else:
                next_x_train = np.vstack((next_x_train, self.X_train[self.client_indices_list[split_ind]]))

            if next_y_train is None:
                next_y_train = self.train_label[self.client_indices_list[split_ind]]
            else:
                next_y_train = np.hstack((next_y_train, self.train_label[self.client_indices_list[split_ind]]))

        inputs_train = torch.from_numpy(next_x_train)
        labels_train = torch.from_numpy(next_y_train)
        inputs_test = torch.from_numpy(next_x_test)
        labels_test = torch.from_numpy(next_y_test)

        return (inputs_train, labels_train), (inputs_test, labels_test)


# Federated Heterogenous MNIST classification: 100 clients, each having 2 classes only (clients have 600 datapoints)
# num_clients: total number of clients (<=100) to consider
# fashionmnist: if FashionMNIST dataset (and not MNIST)
class FederatedHeterogenousMNISTDataGenerator():
    def __init__(self, num_clients=100, seed=0, fashionmnist=False, path=None):

        self.state = {}  # Everything required for recreating this data_generator (can save this in a file and load it)

        # Load self.state from path
        if path is not None:
            self.state = torch.load(path)['data_generator']
            num_clients = self.state['num_clients']
            fashionmnist = self.state['fashionmnist']

        # Generate new data
        else:
            self.state['seed'] = seed
            self.state['num_clients'] = num_clients
            self.state['fashionmnist'] = fashionmnist

        if fashionmnist:
            train_set = datasets.FashionMNIST('./data', download=True, train=True)
            test_set = datasets.FashionMNIST('./data', download=True, train=False)
            self.X_train = train_set.data.numpy().astype(np.float32)
            self.X_train = self.X_train.reshape(self.X_train.shape[0],-1)
            self.train_label = train_set.targets.numpy()
            self.X_test = test_set.data.numpy().astype(np.float32)
            self.X_test = self.X_test.reshape(self.X_test.shape[0],-1)
            self.test_label = test_set.targets.numpy()
        else:
            # train_set = datasets.MNIST('data', download=False, train=True)
            # test_set = datasets.MNIST('data', download=False, train=False)
            # self.X_train = train_set.data.numpy().astype(np.float32)
            # self.X_train = self.X_train.reshape(self.X_train.shape[0],-1)
            # self.train_label = train_set.targets.numpy()
            # self.X_test = test_set.data.numpy().astype(np.float32)
            # self.X_test = self.X_test.reshape(self.X_test.shape[0],-1)
            # self.test_label = test_set.targets.numpy()
            f = gzip.open('data/mnist.pkl.gz', 'rb')
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            f.close()
            self.X_train = np.vstack((train_set[0], valid_set[0]))
            self.X_test = test_set[0]
            self.train_label = np.hstack((train_set[1], valid_set[1]))
            self.test_label = test_set[1]

        self.binary_classification = False  # If False, this is multinomial classification
        self.without_trainloaders = True  # If True, do not use trainloaders

        self.dimensions = self.X_train.shape[1]
        self.num_parameters = self.dimensions
        self.num_outputs = 10

        # This holds the indices of the datapoints for each client, split by class
        self.class_client_splits = {}
        self.class_client_splits_number = {}  # This holds the number of datapoints from each class to store in each client
        for client_ind in range(num_clients):
            self.class_client_splits[client_ind] = {}
            self.class_client_splits_number[client_ind] = {}
            for class_ind in range(10):
                self.class_client_splits[client_ind][class_ind] = []
                self.class_client_splits_number[client_ind][class_ind] = 0

        # Each class is split into 20 parts, with each part assigned to a client
        num_times_class_assigned = np.zeros((10))
        for client_ind in range(num_clients):
            classes = [-1, -1]

            # First class in this client is the one sampled least so far
            classes[0] = np.argmin(num_times_class_assigned)
            num_times_class_assigned[classes[0]] += 1

            # Sample the second class for this client, making sure it is not the same as the first class,
            # and it has not been sampled more than 20 times before
            classes[1] = np.random.choice(10, 1, replace=False)[0]
            while classes[0] == classes[1] or num_times_class_assigned[classes[1]] >= 20:
                classes[1] = np.random.choice(10, 1, replace=False)[0]
            num_times_class_assigned[classes[1]] += 1

            # Assign the 2 clients to class_client_splits_number
            self.class_client_splits_number[client_ind][classes[0]] = 300
            self.class_client_splits_number[client_ind][classes[1]] = 300

        # Assign the data to each client, split by how much data we want from each class
        for class_ind in range(10):
            indices_in_class = np.where(self.train_label == class_ind)[0]
            np.random.shuffle(indices_in_class)  # Shuffle dataset

            num_assigned_so_far = 0
            for client_ind in range(num_clients):
                num_assign_now = self.class_client_splits_number[client_ind][class_ind]
                self.class_client_splits[client_ind][class_ind] = indices_in_class[num_assigned_so_far:num_assigned_so_far + num_assign_now]
                if len(self.class_client_splits[client_ind][class_ind]) != num_assign_now:
                    # Repeat data if not enough data available in class
                    self.class_client_splits[client_ind][class_ind] = indices_in_class[-num_assign_now:]
                num_assigned_so_far += num_assign_now
                print(client_ind, class_ind, num_assign_now)

        # This holds all the indices for each client
        self.client_indices_list = {}
        self.num_points_per_client = []
        for client_ind in range(num_clients):
            self.client_indices_list[client_ind] = np.array([], dtype=np.int64)
            for class_ind in range(10):
                self.client_indices_list[client_ind] = np.concatenate((self.client_indices_list[client_ind],
                                                                       self.class_client_splits[client_ind][class_ind]))

            self.num_points_per_client.append(len(self.client_indices_list[client_ind]))


    # Return trainloader and testloader for a specific split_ind
    def data_split(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            next_x_train = self.X_train[self.client_indices_list[client_ind]]
            next_y_train = self.train_label[self.client_indices_list[client_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            return (inputs_train, labels_train), (inputs_test, labels_test)

    # Do not return TensorDatasets, rather just the tensors, for training data (useful for removing specific points)
    def data_split_arrays_only(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:
            next_x_train = self.X_train[self.client_indices_list[client_ind]]
            next_y_train = self.train_label[self.client_indices_list[client_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            return (inputs_train, labels_train), TensorDataset(inputs_test, labels_test)


    # Return all data in dataloader
    def full_data(self, task_id=None, inputs_only=False):

        if task_id is None:
            task_id = self.state['num_clients']-1

        next_x_train = None
        next_y_train = None
        next_x_test = None
        next_y_test = None

        for split_ind in range(task_id+1):
            # next_x_train = self.X_train[self.client_indices_list[split_ind]]
            # next_y_train = self.train_label[self.client_indices_list[split_ind]]
            next_x_test = self.X_test
            next_y_test = self.test_label

            if next_x_train is None:
                next_x_train = self.X_train[self.client_indices_list[split_ind]]
            else:
                next_x_train = np.vstack((next_x_train, self.X_train[self.client_indices_list[split_ind]]))

            if next_y_train is None:
                next_y_train = self.train_label[self.client_indices_list[split_ind]]
            else:
                next_y_train = np.hstack((next_y_train, self.train_label[self.client_indices_list[split_ind]]))

        inputs_train = torch.from_numpy(next_x_train)
        labels_train = torch.from_numpy(next_y_train)
        inputs_test = torch.from_numpy(next_x_test)
        labels_test = torch.from_numpy(next_y_test)

        return (inputs_train, labels_train), (inputs_test, labels_test)



# Federated CIFAR10 classification: K clients
# num_clients: total number of data splits (=K) to consider
# dataset_proportion: if < 1, how much of dataset to include in setup (eg 0.1)
# data_split: 'iid' or 'non_iid' (homog vs heterog split of data)
# heterog_alpha_1: If 'non_iid', Dir(alpha_1) is num points per client (high == equal split)
# heterog_alpha_2: If 'non_iid', Dir(alpha_2) is num points per class within each client (high == equal split)
class FederatedCIFAR10DataGenerator():
    def __init__(self, num_clients=10, seed=0, dataset_proportion=1., data_split='non_iid',
                 heterog_alpha_1=1, heterog_alpha_2=0.5, cifar100=False, path=None):

        self.state = {}  # Everything required for recreating this data_generator (can save this in a file and load it)

        # Load self.state from path
        if path is not None:
            self.state = torch.load(path)['data_generator']
            num_clients = self.state['num_clients']
            dataset_proportion = self.state['dataset_proportion']
            cifar100 = self.state['cifar100']

        # Generate new data
        else:
            self.state['seed'] = seed
            self.state['num_clients'] = num_clients
            self.state['dataset_proportion'] = dataset_proportion
            self.state['cifar100'] = cifar100

        # Get the CIFAR10 training data
        # Normalise training set together with augmentation
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        # Normalise test set same as training set without augmentation
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

        if cifar100:
            # Load the CIFAR100 dataset
            self.train_dataset_full = datasets.CIFAR100('./data', download=True, train=True, transform=train_transform)
            self.val_dataset_full = datasets.CIFAR100('./data', download=True, train=False, transform=test_transform)
        else:
            # Load the CIFAR10 dataset
            self.train_dataset_full = datasets.CIFAR10('./data', download=True, train=True, transform=train_transform)
            self.val_dataset_full = datasets.CIFAR10('./data', download=True, train=False, transform=test_transform)

        self.binary_classification = False  # If False, this is multinomial classification
        self.without_trainloaders = False  # If True, do not use trainloaders

        self.dimensions = 3*32*32#self.X_train.shape[1]
        self.num_parameters = self.dimensions
        self.num_outputs = 10
        if cifar100:
            self.num_outputs = 100

        # This holds the indices of the datapoints for each client, split by class
        self.class_client_splits = {}
        self.class_client_splits_number = {}  # This holds the number of datatpoints from each class to store in each client
        for client_ind in range(num_clients):
            self.class_client_splits[client_ind] = {}
            self.class_client_splits_number[client_ind] = {}
            for class_ind in range(self.num_outputs):
                self.class_client_splits[client_ind][class_ind] = []
                self.class_client_splits_number[client_ind][class_ind] = []

        # Assign how much data we want from each class in each client (fill self.class_client_splits_number)
        if data_split == 'iid':
            alpha_1 = 1000000000000000
            alpha_2 = 1000000000000000
        elif data_split == 'non_iid':
            alpha_1 = heterog_alpha_1  # Uneven distribution of number of points per client
            alpha_2 = heterog_alpha_2  # Uneven distribution of classes within each client. Can try alpha_2<1 too (eg 0.5 or 0.3 or 0.1)
        elif data_split == 'non_iid_1_05':
            alpha_1 = 1  # Uneven distribution of number of points per client
            alpha_2 = 0.5  # Uneven distribution of classes within each client. Can try alpha_2<1 too (eg 0.5 or 0.3 or 0.1)
        else:
            alpha_1 = 1000000000000000  # Dir(alpha_1) is num points per client (high == equal split)
            alpha_2 = 1000000000000000  # Dir(alpha_2) is num points per class within each client (high == equal split)
            print("Did not set data_split variable correctly in the data generator. Assuming iid split / distribution.")

        client_splits = np.random.dirichlet([alpha_1 for _ in range(num_clients)])
        client_class_splits = np.random.dirichlet([alpha_2 for _ in range(self.num_outputs)], (num_clients))
        normaliser_when_assigning_number = np.zeros((self.num_outputs))
        for client_ind in range(num_clients):
            self.class_client_splits_number[client_ind] = client_splits[client_ind] * client_class_splits[client_ind]
            for class_ind in range(self.num_outputs):
                normaliser_when_assigning_number[class_ind] += self.class_client_splits_number[client_ind][class_ind]

        # Assign the data to each client, split by how much data we want from each class
        for class_ind in range(self.num_outputs):
            # indices_in_class = np.where(self.train_label == class_ind)[0]
            indices_in_class = torch.tensor(self.train_dataset_full.targets) == class_ind
            indices_in_class = indices_in_class.nonzero().reshape(-1)
            num_in_class = len(indices_in_class) * dataset_proportion  # Only use a proportion of the dataset
            random_shuffle = torch.randperm(len(indices_in_class))  # Shuffle dataset
            indices_in_class = indices_in_class[random_shuffle]
            # np.random.shuffle(indices_in_class)  # Shuffle dataset

            num_assigned_so_far = 0
            for client_ind in range(num_clients):
                num_assign_now = np.round(num_in_class * self.class_client_splits_number[client_ind][class_ind] / normaliser_when_assigning_number[class_ind]).astype(int)
                if num_assign_now == 1:
                    num_assign_now = 2
                self.class_client_splits[client_ind][class_ind] = indices_in_class[num_assigned_so_far:num_assigned_so_far + num_assign_now]
                num_assigned_so_far += num_assign_now
                print(client_ind, class_ind, num_assign_now)

        # This holds all the indices for each client
        self.client_indices_list = {}
        self.num_points_per_client = []
        for client_ind in range(num_clients):
            self.client_indices_list[client_ind] = np.array([], dtype=np.int64)
            for class_ind in range(self.num_outputs):
                self.client_indices_list[client_ind] = np.concatenate((self.client_indices_list[client_ind],
                                                                       self.class_client_splits[client_ind][class_ind]))

            self.num_points_per_client.append(len(self.client_indices_list[client_ind]))


    # Return trainloader and testloader for a specific split_ind
    def data_split(self, client_ind):
        if client_ind >= self.state['num_clients']:
            raise Exception("Number of data splits exceeded!")
        else:

            from torch.utils.data import Subset
            train_subset = Subset(self.train_dataset_full, self.client_indices_list[client_ind])

            return train_subset, self.val_dataset_full

    # Return all data in dataloader
    def full_data(self, task_id=None):
        # if task_id is None:
        #     task_id = self.state['num_splits']-1

        all_inds = np.array([], dtype=np.int64)
        for client_ind in self.client_indices_list:
            all_inds = np.concatenate((all_inds, self.client_indices_list[client_ind]))

        full_train_data = Subset(self.train_dataset_full, all_inds)

        return full_train_data, self.val_dataset_full

