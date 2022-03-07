from pdb import Restart
import six
import sys
sys.modules['sklearn.externals.six'] = six

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlrose
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import time
import matplotlib.pyplot as plt
from mlrose import GeomDecay
from sklearn.datasets import load_breast_cancer


#Random State
rs = 614


def _plot_helper_(x_axis, train_scores, test_scores, train_label, test_label, name):
    """Plot helper.

        Args:
            x_axis (ndarray): x axis array.
            train_scores (ndarray): array of training scores.
            test_scores (ndarray): array of validation scores.
            train_label (string): training plot label.
            test_label (string): validation plot label.

        Returns:
            None.
        """


    # Plot training and validation mean by filling in between mean + std and mean - std
    plt.plot(x_axis, train_scores, markersize=2, label=train_label)
    plt.plot(x_axis, test_scores,  markersize=2, label=test_label)
    plt.title('Accuracy of {}'.format(name))
    plt.legend(loc='lower right')
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    plt.savefig('images\{}_learning_curve'.format(name))
    plt.close()

def rhc(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose.random_hill_climb(problem_fit, max_attempts=100, max_iters= max_iters, restarts=10, random_state = rs)[1]
    return [max_iters, "random_hill_climb", problem_name,fitness_score, time.time()-start]

def sa(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose.simulated_annealing(problem_fit, max_attempts=100, max_iters= max_iters, random_state = rs)[1]
    return [max_iters, "simulated_annealing", problem_name,fitness_score, time.time()-start]

def ga(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose.genetic_alg(problem_fit, max_attempts=100, max_iters= max_iters, pop_size= 200, mutation_prob=0.1, random_state = rs)[1]
    return [max_iters, "genetic_alg", problem_name,fitness_score, time.time()-start]

def mimic(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose.mimic(problem_fit, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=max_iters, curve=False, random_state=rs, fast_mimic=True)[1]
    return [max_iters, "mimic", problem_name,fitness_score, time.time()-start]



if __name__ == "__main__":

    data = load_breast_cancer()
    x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
   
    x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.2, shuffle=True,
                                                        random_state=42, stratify=y)

    
    # x1_data,y1_data = dataset.dataAllocation('pima-indians-diabetes.csv')
    # x1_train, x1_test, y1_train, y1_test = dataset.trainSets(x1_data,y1_data)

    scaler = StandardScaler()
    scaled_x1_train = scaler.fit_transform(x1_train)
    scaled_x1_test = scaler.transform(x1_test)

    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']


### PART 1 ###
    

    for algorithm in algorithms:
            
        results = []
        test_score = []
        train_score = []
        x_axis = []
        
        """ Variables to change located here"""
        plot_name = "max_iter"
        max_range = 500
        plot_name = str(algorithm + "_" + plot_name)

        for var in range(0, max_range, 50):
        # for var in ['mlrose.GeomDecay()','mlrose.ArithDecay()','mlrose.ExpDecay()']:
            model = mlrose.NeuralNetwork(hidden_nodes=[4], activation='relu',
                                            algorithm=algorithm, max_iters=var,
                                            bias=True, is_classifier=True, learning_rate=0.829,
                                            early_stopping=True, clip_max=1,
                                            pop_size=600,
                                            random_state=rs)
            """ Variables to change end """                        
            model.fit(scaled_x1_train, y1_train)
            y_train_pred = model.predict(scaled_x1_train)
            y_train_accuracy = accuracy_score(y1_train, y_train_pred)


            y_test_pred = model.predict(scaled_x1_test)
            y_test_accuracy = accuracy_score(y1_test, y_test_pred)

            f1score = f1_score(y1_test, y_test_pred)

            results.append([var, algorithm, y_train_accuracy, y_test_accuracy, f1score])
            x_axis.append(var)
            test_score.append(y_test_accuracy)
            train_score.append(y_train_accuracy)

        max_index = np.argmax(test_score)
        print('Max accuracy: ' + str(results[max_index]))
        print(results)

        _plot_helper_(x_axis, train_score, test_score, train_label='Train set', test_label='Test set', name=plot_name)
    






### PART 2 ###
    results = []
    problems_name = ["Flip Flop", "One Max", "FourPeaks"] 
    fitness_functions = [mlrose.FlipFlop(), mlrose.OneMax(), mlrose.FourPeaks()] 
    
    problems = [mlrose.DiscreteOpt(length = 100, fitness_fn = fitness_function, maximize=True, max_val = 2) for fitness_function in fitness_functions]
    
    for j in range(len(problems)):
        for i in range(0, 1000, 100): #From 2500
            results.append(rhc(problems[j], problems_name[j], max_iters= i))
            results.append(sa(problems[j], problems_name[j], max_iters= i))
            results.append(ga(problems[j], problems_name[j], max_iters= i))
            results.append(mimic(problems[j], problems_name[j], max_iters= i))
            print(i, end=" ")

    df = pd.DataFrame(results, columns=["Iteration", "Algorithm", "Problem", "Fitness", "Time"])
    sns.lineplot(data=df[df['Problem']==problems_name[1]], x="Iteration", y="Fitness", hue="Algorithm")
    df.to_csv("problems_solution.csv", index=False)


    for problem in problems_name:
        plt.figure()
        sns.lineplot(data=df[df['Problem']==problem], x="Iteration", y="Fitness", hue="Algorithm").set_title(problem+ ": Fitness vs Iterations 1")
        plt.savefig(f'images\{problem}_Fitness_Iterations')
        plt.close()
        plt.figure()
        sns.lineplot(data=df[df['Problem']==problem], x="Iteration", y="Time", hue="Algorithm").set_title(problem+  ": Time vs Iterations 2")
        plt.savefig(f'images\{problem}_Time_Iterations')
        plt.close()

    df.groupby(['Algorithm', 'Problem'])['Fitness'].max().to_csv("problems_fitness3.csv", index=False)

    df.groupby(['Algorithm', 'Problem'])['Time'].max().to_csv("problems_time4.csv", index=False)

    df[df['Problem']=='Flip Flop'].groupby(['Algorithm', 'Problem'])['Time'].mean().to_csv("FlipFlop_time5.csv", index=False)

    df[df['Problem']=='One Max'].groupby(['Algorithm', 'Problem'])['Time'].mean().to_csv("OneMax_time6.csv", index=False)

    df[df['Problem']=='FourPeaks'].groupby(['Algorithm', 'Problem'])['Time'].mean().to_csv("MaxK_time7.csv", index=False)
    plt.figure()
    sns.lineplot(data=df[(df['Problem']=='FourPeaks') & (df['Iteration'] < 1000)], x="Iteration", y="Fitness", hue="Algorithm").set_title(problem+ ": Fitness vs Iterations")
    plt.savefig(f'images\{problem}_fitness_iterations8')
    plt.close()
    plt.figure()
    sns.lineplot(data=df[(df['Problem']=='FourPeaks') & (df['Iteration'] < 1000)], x="Iteration", y="Time", hue="Algorithm").set_title(problem+ ": Time vs Iterations")
    plt.savefig(f'images\{problem}_time_iterations9')
    plt.close()

    
