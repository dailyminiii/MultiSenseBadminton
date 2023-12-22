############
# MIT License
#
# Copyright (c) 2023 Minwoo Seong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from sklearn.preprocessing import LabelEncoder
from preprocessing import BadmintonDataset
from sklearn.model_selection import KFold
from Networks.SciDataModels import Conv1DRefinedModel, ConvLSTMRefinedModel, LSTMRefinedModel, TransformerRefinedModel
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
import argparse
import h5py
import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(labels, predictions, task_name, classes, savefolder, normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'Confusion matrix for {task_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(savefolder)
    plt.clf()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save(model, config, saved_folder_name, fold=None, epoch=None):
    output_folder_name = saved_folder_name
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        config_name = output_folder_name + '/config_'+ 'S_' + str(fold)
        model_name = output_folder_name + '/model_' + 'S_' + str(fold)
    else:        
        config_name = output_folder_name + '/config_'+ 'S_' + str(fold) + '_Ep_' + str(epoch)
        model_name = output_folder_name + '/model_' + 'S_' + str(fold) + '_Ep_' + str(epoch)

    torch.save(model.state_dict(), model_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))

def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--output_folder_name",
                     type=str,
                     help="path to save model",
                     default='test_2')
    opt.add_argument("--seed_value",
                     type=int,
                     default=42,
                     help="seed value")
    opt.add_argument("--batch_size",
                     type=int,
                     default=64,
                     help="batch size")
    opt.add_argument("--num_folds",
                     type=int,
                     default=10,
                     help="Number of fold for dataset")
    opt.add_argument("--patience",
                     type=int,
                     default=15,
                     help="Number of patience for early stopping")
    opt.add_argument("--gpu_num",
                     type=int,
                     default=5,
                     help="Selected GPU number")
    config = vars(opt.parse_args())
    return config

random_seed = 42
set_random_seed(random_seed)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define the preprocessed data path
DataPath = './data_processed/data_processed_allStreams_60hz_allActs.hdf5'

# Data and Hyperparameters Setting
patience = 10  # Number of epochs to wait for improvement
num_epochs = 200  # Change this to the desired number of epochs

batch_size_list = [128]
hidden_features_list = [32, 64]
learning_rate_list = [0.0005, 0.0001]

sensor_subset_list = [
                        'allStreams',
                        'noGforce',
                        'noCognionics',
                        'noEye',
                        'noInsole',
                        'noBody',
                        'onlyGforce',
                        'onlyCognionics',
                        'onlyEye',
                        'onlyInsole',
                        'onlyBody'
                        ]

ground_truth_list = [
    'stroke_type',
    'skill_level',
    'hori',
    'ver',
    'hiting',
    'sound',
]

model_list = [
    'ConvLSTM',
    'Transformer',
    'Conv1D',
    'LSTM',
]

for modelName in model_list:
    for gound_truth in ground_truth_list:
        for batch_size in batch_size_list:
            for hidden_features in hidden_features_list:
                for learning_rate in learning_rate_list:
                    for subset in sensor_subset_list:
                        config = get_argument()
                        config['lr'] = learning_rate
                        config['batch_size'] = batch_size
                        config['patience'] = patience
                        config['max_epochs'] = num_epochs
                        config['hidden_features'] = hidden_features
                        config['sensor_subset'] = subset
                        config['ground_truth'] = gound_truth
                        config['modelName'] = modelName

                        # Load Experimental Dataset
                        feature_matrices = h5py.File(DataPath, 'r')['example_matrices'][:]
                        feature_matrices_subject_ids = h5py.File(DataPath, 'r')['example_subject_ids'][:]
                        feature_matrices_subject_ids_str = np.array([x.decode('utf-8') for x in feature_matrices_subject_ids])
                        feature_matrices_subject_encoded_labels = LabelEncoder().fit_transform(feature_matrices_subject_ids_str)

                        feature_matrices_stroke_type = h5py.File(DataPath, 'r')['example_label_indexes'][:]
                        feature_matrices_skill_level = h5py.File(DataPath, 'r')['example_skill_level'][:]

                        feature_matrices_hori_score = h5py.File(DataPath, 'r')['example_score_annot_3_hori'][:]
                        feature_matrices_ver_score = h5py.File(DataPath, 'r')['example_score_annot_3_ver'][:]
                        feature_matrices_hitting_score = h5py.File(DataPath, 'r')['example_score_annot_4'][:]
                        feature_matrices_sound_score = h5py.File(DataPath, 'r')['example_score_annot_5'][:]

                        # Forehand Indexing
                        forehand_indices = np.where(feature_matrices_stroke_type == 1)[0]

                        forehand_feature_matrices = feature_matrices[forehand_indices]
                        forehand_feature_matrices_subject_ids_str =feature_matrices_subject_ids_str[forehand_indices]
                        forehand_feature_matrices_annot_2_skill_level = feature_matrices_skill_level[forehand_indices]
                        forehand_feature_matrices_annot_3_hori_score = feature_matrices_hori_score[forehand_indices]
                        forehand_feature_matrices_annot_3_ver_score = feature_matrices_ver_score[forehand_indices]
                        forehand_feature_matrices_annot_4_hitting_score = feature_matrices_hitting_score[forehand_indices]
                        forehand_feature_matrices_annot_5_sound_score = feature_matrices_sound_score[forehand_indices]

                        if gound_truth == 'stroke_type':
                            feature_matrices_ground_truth = feature_matrices_stroke_type
                            label_num = 3
                        elif gound_truth == 'skill_level':
                            feature_matrices = forehand_feature_matrices
                            feature_matrices_ground_truth = forehand_feature_matrices_annot_2_skill_level
                            label_num = 3
                        elif gound_truth == 'hori':
                            feature_matrices = forehand_feature_matrices
                            feature_matrices_ground_truth = forehand_feature_matrices_annot_3_hori_score
                            label_num = 6
                        elif gound_truth == 'ver':
                            feature_matrices = forehand_feature_matrices
                            feature_matrices_ground_truth = forehand_feature_matrices_annot_3_ver_score
                            label_num = 7
                        elif gound_truth == 'hitting':
                            feature_matrices = forehand_feature_matrices
                            feature_matrices_ground_truth = forehand_feature_matrices_annot_4_hitting_score
                            label_num = 3
                        elif gound_truth == 'sound':
                            feature_matrices = forehand_feature_matrices
                            feature_matrices_ground_truth = forehand_feature_matrices_annot_5_sound_score
                            label_num = 3
                       

                        hyper = './hyper_LOSO_' + modelName + '/bat' + str(batch_size) + "hid" + str(hidden_features) + "lr" +str(learning_rate) + 'gt_' + gound_truth + '_sensor_' + subset
                        print(hyper)
                        config['output_folder_name'] = hyper

                        # Check if GPU is available
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        print(device)
                        if not os.path.exists(config['output_folder_name']):
                            os.makedirs(config['output_folder_name'])

                        feature_idx = 0

                        if subset == 'allStreams':
                            input_feature_matrices = feature_matrices
                        elif subset == 'noGforce':
                            input_feature_matrices = np.concatenate((feature_matrices[:, :, :2], feature_matrices[:, :, 18:]), axis=2)

                        elif subset == 'noCognionics':
                            input_feature_matrices = np.concatenate((feature_matrices[:, :, :18], feature_matrices[:, :, 22:]), axis=2)

                        elif subset == 'noEye':
                            input_feature_matrices = feature_matrices[:, :, 2:]

                        elif subset == 'noInsole':
                            input_feature_matrices = np.concatenate((feature_matrices[:, :, :22], feature_matrices[:, :, 58:]), axis=2)

                        elif subset == 'noBody':
                            input_feature_matrices = feature_matrices[:, :, :58]

                        elif subset == 'onlyGforce':
                            input_feature_matrices = feature_matrices[:, :, 2:18]

                        elif subset == 'onlyEye':
                            input_feature_matrices = feature_matrices[:, :, 0:2]

                        elif subset == 'onlyInsole':
                            input_feature_matrices = feature_matrices[:, :, 22:58]

                        elif subset == 'onlyBody':
                            input_feature_matrices = feature_matrices[:, :, 58:]

                        base_dir = hyper + '/'

                        train_acc_list_k = []
                        test_acc_list_k = []
                        train_loss_list_k = []
                        test_loss_list_k = []

                        balanced_acc_train_k, f1_weighted_train_k = [], []
                        balanced_acc_test_k, f1_weighted_test_k = [], []

                        dataset = BadmintonDataset(input_feature_matrices, feature_matrices_ground_truth)

                        kfold = KFold(n_splits=config['num_folds'], shuffle=True)

                        for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
                            print(f"Fold {fold + 1}/{config['num_folds']}")

                            saved_folder_name = base_dir + 'K' + str(fold + 1)

                            os.makedirs(saved_folder_name, exist_ok=True)

                            # Create subsets for training, validation, and testing
                            train_dataset = torch.utils.data.Subset(dataset, train_indices)
                            test_dataset = torch.utils.data.Subset(dataset, test_indices)

                            if gound_truth == 'stroke_type':

                                def count_class_samples(dataset, indices):
                                    class_counts = Counter()
                                    for idx in indices:
                                        _, label_tensor = dataset[idx]
                                        label = label_tensor.item()  # Convert tensor to a Python scalar
                                        class_counts[label] += 1
                                    return class_counts

                                # Count the number of samples for each class in the train and test datasets
                                train_class_counts = count_class_samples(dataset, train_indices)
                                test_class_counts = count_class_samples(dataset, test_indices)

                                print("Number of samples per class in the training dataset:", train_class_counts)
                                print("Number of samples per class in the test dataset:", test_class_counts)

                                def reduce_label_samples_by_half(dataset, indices, label_to_reduce):
                                    label_indices = [idx for idx in indices if dataset[idx][1].item() == label_to_reduce]
                                    reduced_label_indices = random.sample(label_indices, len(label_indices) // 2)
                                    reduced_indices = [idx for idx in indices if idx not in reduced_label_indices]

                                    return reduced_indices

                                reduced_train_indices = reduce_label_samples_by_half(dataset, train_indices, 2)
                                reduced_test_indices = reduce_label_samples_by_half(dataset, test_indices, 2)

                                reduced_train_dataset = torch.utils.data.Subset(dataset, reduced_train_indices)
                                reduced_test_dataset = torch.utils.data.Subset(dataset, reduced_test_indices)

                                reduced_train_class_counts = count_class_samples(dataset, reduced_train_indices)
                                reduced_test_class_counts = count_class_samples(dataset, reduced_test_indices)

                                print("Number of samples per class in the reduced training dataset:", reduced_train_class_counts)
                                print("Number of samples per class in the reduced test dataset:", reduced_test_class_counts)

                                train_dataset = reduced_train_dataset
                                test_dataset = reduced_test_dataset

                            # Initialize counters for each type of annotation
                            traindata_distribution= Counter()
                            testdata_distribution = Counter()

                            # Iterate through the dataset and update the counters
                            for _, label_dict in train_dataset:
                                traindata_distribution.update([label_dict.item()])

                            print("Distribution of Horizontal Score Labels:", traindata_distribution)
                            print()

                            # Training loop
                            train_losses = []  
                            train_acc = []
                            test_losses = []  
                            test_acc = []

                            balanced_acc_train_list, f1_weighted_train_list = [], []
                            balanced_acc_test_list, f1_weighted_test_list = [], []

                            # Initialize lists to store predictions and true labels for each task
                            train_true_labels_KAVG = []
                            train_predicted_probs_KAVG = []
                            test_true_labels_KAVG = []
                            test_predicted_probs_KAVG = []

                            print('input size : ', len(input_feature_matrices[0,0,:]))

                            if modelName == "Conv1D":
                                model = Conv1DRefinedModel(len(input_feature_matrices[0,0,:]), hidden_features, output_size=label_num).to(device)
                            elif modelName == "LSTM":
                                model = LSTMRefinedModel(len(input_feature_matrices[0,0,:]), hidden_features, output_size=label_num).to(device)
                            elif modelName == "ConvLSTM":
                                model = ConvLSTMRefinedModel(len(input_feature_matrices[0,0,:]), hidden_features, output_size=label_num).to(device)
                            elif modelName == "Transformer":
                                model = TransformerRefinedModel(len(input_feature_matrices[0,0,:]), hidden_features, output_size=label_num).to(device)


                            # Loss function and optimizer setup
                            criterion_cf = nn.CrossEntropyLoss()  # Mean Squared Error loss
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

                            # Create data loaders for training and validation
                            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        
                            # Define early stopping parameters
                        
                            min_delta = 0.001  # Minimum change in validation loss to be considered as an improvement
                            best_loss = float('inf')  # Initialize the best validation loss
                            epochs_without_improvement = 0  # Initialize the counter for epochs without improvement

                            for epoch in tqdm(range(num_epochs)):
                                running_loss = 0.0
                                correct_instance, total_instance = 0, 0

                                # Initialize lists to store predictions and true labels for each task
                                train_true_labels = []
                                train_predicted_probs = []
                                test_true_labels = []
                                test_predicted_probs = []

                                model.train()
                                for inputs, targets in train_loader:
                                
                                    optimizer.zero_grad()  # Zero gradients to prevent accumulation
                                    outputs = model(inputs.to(device))  # Forward pass through the multi-task model

                                    loss = criterion_cf(outputs, targets.long().to(device))                           

                                    loss.backward()  # Backpropagation to compute gradients
                                    optimizer.step()  # Update the model parameters

                                    running_loss += loss.item()
                                
                                    _, predicted_instance = torch.max(outputs, 1)
                                    total_instance += targets.size(0)
                                    correct_instance += (predicted_instance == targets.long().to(device)).sum().item()
                
                                    train_true_labels.extend(targets.cpu().numpy())
                                    train_predicted_probs.extend(predicted_instance.cpu().detach().numpy())

                                    # K AVG
                                    train_true_labels_KAVG.extend(targets.cpu().numpy())
                                    train_predicted_probs_KAVG.extend(predicted_instance.cpu().detach().numpy())

                                # Test loop
                                model.eval()  # Set the model to evaluation mode
                                running_test_loss = 0.0        
                                correct_instance_test, total_instance_test = 0, 0

                                with torch.no_grad():  # No gradient calculation during testing
                                    for test_inputs, test_targets in test_loader:
                                        test_outputs = model(test_inputs.to(device))

                                        test_loss = criterion_cf(test_outputs, test_targets.long().to(device))
                                        
                                        running_test_loss += test_loss.item()
                                    
                                        _, predicted_instance_test = torch.max(test_outputs, 1)
                                        total_instance_test += test_targets.size(0)
                                        correct_instance_test += (predicted_instance_test == test_targets.long().to(device)).sum().item()

                                        test_true_labels.extend(test_targets.cpu().numpy())
                                        test_predicted_probs.extend(predicted_instance_test.cpu().detach().numpy())


                                bal_acc_train = balanced_accuracy_score(train_true_labels,
                                                                                train_predicted_probs)
                                f1_weighted_train = f1_score(train_true_labels,
                                                                train_predicted_probs,
                                                                average='weighted')

                                # Balanced accuracy & F1 weighted score
                                bal_acc_test = balanced_accuracy_score(test_true_labels,
                                                                        test_predicted_probs)
                                f1_weighted_test = f1_score(test_true_labels,
                                                                test_predicted_probs,
                                                                average='weighted')

                                
                                # K AVG

                                test_true_labels_KAVG.extend(test_targets.cpu().numpy())
                                test_predicted_probs_KAVG.extend(predicted_instance_test.cpu().detach().numpy())
                                    
                                        
                                print()
                                print(f"Epoch {epoch+1},  Train Accuracy: {100 * correct_instance / total_instance}%  total : {total_instance}  correct : {correct_instance}   ")
                                print(f"        Test Accuracy: {100 * correct_instance_test / total_instance_test}%  total : {total_instance_test}  correct : {correct_instance_test}   ")
                                print(f"        Train Balanced Accuracy: {100 * bal_acc_train}%")
                                print(f"        Test Balanced Accuracy: {100 * bal_acc_test}%")
                                print(f"        Train F1 Accuracy: {100 * f1_weighted_train}%")
                                print(f"        Test F1 Accuracy: {100 * f1_weighted_test}%")
                                print(f"        Train Loss: {running_loss/len(train_loader)}")
                                print(f"        Test Loss: {running_test_loss/len(test_loader)}")
                                print(f"        Best Loss: {best_loss}")
                                print(f"        Epochs_without_improvement: {epochs_without_improvement}")
                                print()            
                                print()

                                train_losses.append(running_loss/len(train_loader))
                                train_acc.append(correct_instance / total_instance)
                                test_losses.append(running_test_loss/len(test_loader))
                                test_acc.append(correct_instance_test / total_instance_test)

                                balanced_acc_train_list.append(bal_acc_train)
                                f1_weighted_train_list.append(f1_weighted_train)
                                balanced_acc_test_list.append(bal_acc_test)
                                f1_weighted_test_list.append(f1_weighted_test)

                                # Check for early stopping
                                if (running_test_loss / len(test_loader)) + min_delta < best_loss:
                                    best_loss = running_test_loss / len(test_loader)
                                    epochs_without_improvement = 0
                                else:
                                    epochs_without_improvement += 1
                                
                                if epochs_without_improvement >= patience:
                                    print("Early stopping: Validation loss has not improved for the last", patience, "epochs.")
                                    break

                            train_acc_list_k.append(train_acc)                        
                            test_acc_list_k.append(test_acc)                    
                            train_loss_list_k.append(train_losses)
                            test_loss_list_k.append(test_losses)

                            balanced_acc_train_k.append(bal_acc_train)
                            f1_weighted_train_k.append(f1_weighted_train)
                        
                            balanced_acc_test_k.append(bal_acc_test)
                            f1_weighted_test_k.append(f1_weighted_test)
                            
                            # Plot and save the total loss graphs
                            plt.figure(figsize=(8, 8))
                            plt.plot(train_losses, label='Train Loss')
                            plt.plot(test_losses, label='Test Loss')
                            plt.xlabel('Epochs')
                            plt.ylabel('Loss')
                            plt.legend()
                            plt.savefig(saved_folder_name + '/loss.png')
                            plt.clf()

                            # Plot and save the loss graphs
                            plt.figure(figsize=(8, 8))
                            plt.plot(train_acc, label='Train Accuracy')
                            plt.plot(test_acc, label='Test Accuracy')
                            plt.xlabel('Epochs')
                            plt.ylabel('Accuracy')
                            plt.legend()
                            plt.savefig(saved_folder_name+ '/accuracy.png')
                            plt.clf()

                            # Plot and save the loss graphs
                            plt.figure(figsize=(8, 8))
                            plt.plot(balanced_acc_train_list, label='Train Balanced Accuracy')
                            plt.plot(balanced_acc_test_list, label='Test Balanced Accuracy')
                            plt.xlabel('Epochs')
                            plt.ylabel('Balanced Accuracy')
                            plt.legend()
                            plt.savefig(saved_folder_name + '/bal_acc.png')
                            plt.clf()

                            # Plot and save the loss graphs
                            plt.figure(figsize=(8, 8))
                            plt.plot(f1_weighted_train_list, label='Train F1 Accuracy')
                            plt.plot(f1_weighted_test_list, label='Test F1 Accuracy')
                            plt.xlabel('Epochs')
                            plt.ylabel('Intervention F1 Accuracy')
                            plt.legend()
                            plt.savefig(saved_folder_name + '/f1_acc.png')
                            plt.clf()

                            # Using the function to plot the confusion matrix
                            plot_confusion_matrix(train_true_labels, train_predicted_probs, "stroke_type", [0,1,2], saved_folder_name+ '/train_acc_confusion.png')
                            plot_confusion_matrix(test_true_labels, test_predicted_probs, "stroke_type", [0,1,2], saved_folder_name+ '/test_acc_confusion.png')

                            config['train_loss'] = running_loss/len(train_loader)
                            config['train_acc'] = correct_instance / total_instance
                            config['test_loss'] = running_test_loss/len(test_loader)
                            config['test_acc'] = correct_instance_test / total_instance_test

                            config['train_bal_acc'] = 100 * bal_acc_train
                            config['test_bal_acc'] = 100 * bal_acc_test
                            config['train_f1_acc'] = 100 * f1_weighted_train
                            config['test_f1_acc'] = 100 * f1_weighted_test

                            save(model, config, saved_folder_name, fold=None, epoch=epoch+1)
