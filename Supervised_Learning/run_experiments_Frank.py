# Script to run experiments

from pickle import TRUE
from re import I
from tracemalloc import stop
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from adaboost import AdaBoost
from decision_trees import DecisionTree
from k_nearest_neighbors import KNN
from neural_networks import NeuralNetwork
from gradient_boost import GradientBoost
from support_vector_machines import SVM

IMAGE_DIR = 'images/'


def load_dataset(split_percentage=0.2, visualize=False):
    """Load MNIST or WDBC.

       Args:
           split_percentage (float): validation split.
           visualize (bool): True if some of the dataset images have to been shown.

       Returns:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
       """

        # Load WDBC
    data = load_breast_cancer()
    x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
   

    if visualize:

        # Build dataset and assign labels
        df = pd.DataFrame(x, columns=features)
        df['labels'] = y
        df['labels'] = df['labels'].map({1: 'B', 0: 'M'})

        # Plot instances distribution
        plt.figure()
        sns.set(style='darkgrid')

        # sns.countplot(x='labels', data=df, palette={'B': 'b', 'M': 'r'})
        # plt.title('WDBC Instances Distribution')
        # plt.savefig(IMAGE_DIR + 'WDBC_Instances_Distribution')

        # # Plot heatmap of correlations
        # plt.figure(figsize=(15, 15))
        # sns.heatmap(df.corr(), annot=True, square=True, cmap='coolwarm')
        # plt.savefig(IMAGE_DIR + 'WDBC_Features_Correlation')

        # # Plot scatter matrix of features
        # plt.figure(figsize=(15, 15))
        # sns.pairplot(df, hue='labels', palette={'B': 'b', 'M': 'r'})
        # plt.savefig(IMAGE_DIR + 'WDBC_Scatter_Matrix_of_Features')

        # print("Number of features: ", len(features))
        
        # Plot features distributions
        # plt.figure(figsize=(15, 15))

        # for i, feature in enumerate(features):
        #     plt.subplot(15, 2, i + 1)

        #     sns.distplot(x[y == 0, i], color='red', label='M')
        #     sns.distplot(x[y == 1, i], color='blue', label='B')

        #     plt.legend(loc='upper left')
        #     plt.xlabel(feature)

        # # plt.tight_layout()
        # plt.show()
        # plt.savefig(IMAGE_DIR + 'WDBC_Features_Discrimination')
        # plt.close(fig='all')
     

    # Split dataset in training and validation sets, preserving classes representation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True,
                                                        random_state=42, stratify=y)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test


def experiment(x_train, x_test, y_train, y_test):
    """Perform experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.

        Returns:
           None.
        """

    # Array of training sizes to plot the learning curves over.
    training_sizes = np.arange(20, int(len(x_train) * 0.9), 10)

    # K-Nearest Neighbor
    print('\n--------------------------')
    knn = KNN(k=1, weights='uniform', p=2)
    knn.experiment(x_train, x_test, y_train, y_test,
                   cv=10,
                   y_lim=0.3,
                   n_neighbors_range=np.arange(1, 50, 2),
                   p_range=np.arange(1, 20),
                   weight_functions=['uniform', 'distance'],
                   train_sizes=training_sizes)

    # # Support Vector Machines
    print('\n--------------------------')
    # svm = SVM(c=5., kernel='rbf', degree=3, gamma=0.001, random_state=42)
    # svm.experiment(x_train, x_test, y_train, y_test,
    #                cv=10,
    #                y_lim=0.6,
    #                C_range=[1, 5] + list(range(10, 100, 20)) + list(range(100, 1000, 50)),
    #             #    kernels=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #                kernels=['rbf'],
    #                gamma_range=np.logspace(-7, 0, 50),
    #                poly_degrees=[2, 3, 4],
    #                train_sizes=training_sizes
    #                )

    # # Pruned Decision Trees
    # print('\n--------------------------')
    # dt = DecisionTree(max_depth=10, min_samples_leaf=1, ccp_alpha=0.0)
    # dt.experiment(x_train, x_test, y_train, y_test,
    #               cv=10,
    #               y_lim=0.1,
    #               max_depth_range=list(range(1, 50)),
    #               min_samples_leaf_range=list(range(1, 30)),
    #               ccp_alpha_range=np.linspace(0,1,51),
    #               train_sizes=training_sizes)

    # # AdaBoost
    # print('\n--------------------------')
    # boosted_dt = AdaBoost(n_estimators=50, learning_rate=1., max_depth=3, random_state=42)
    # boosted_dt.experiment(x_train, x_test, y_train, y_test,
    #                       cv=10,
    #                       y_lim=0.2,
    #                       max_depth_range=list(range(1, 30)),
    #                       n_estimators_range=[1, 3, 5, 8] + list(range(10, 100, 5)) + list(range(100, 1000, 50)),
    #                       learning_rate_range=np.logspace(-6, 1, 50),
    #                       train_sizes=training_sizes)

    # GradientBoost
    # print('\n--------------------------')
    # boosted_dt = GradientBoost(n_estimators=50, learning_rate=1., max_depth=10, random_state=42)
    # boosted_dt.experiment(x_train, x_test, y_train, y_test,
    #                       cv=10,
    #                       y_lim=0.2,
    #                       max_depth_range=list(range(1, 30)),
    #                       n_estimators_range=[1, 3, 5, 8] + list(range(10, 100, 5)) + list(range(100, 1000, 50)),
    #                       learning_rate_range=np.logspace(-6, 1, 50),
    #                       train_sizes=training_sizes)

    # # Neural Networks
    # print('\n--------------------------')
    # nn = NeuralNetwork(layer1_nodes=17,layer2_nodes=1, learning_rate=0.001, max_iter=100)
    # nn.experiment(x_train, x_test, y_train, y_test,
    #               cv=10,
    #               y_lim=0.1,
    #             #   first_hidden_layer_sizes_range=[(i) for i in range(1,30)],
    #             #   second_hidden_layer_sizes_range=[(17,i) for i in range(1,30)],
    #               learning_rate_range=np.logspace(-4, 0, 50),
    #               train_sizes=training_sizes)


if __name__ == "__main__":

    # Run experiment 1 on WDBC
    print('\n--------------------------')
    x_train, x_test, y_train, y_test = load_dataset(visualize=False)

    experiment(x_train, x_test, y_train, y_test)

