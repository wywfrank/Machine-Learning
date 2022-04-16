# CS7641 - Machine Learning
https://github.com/wywfrank/Machine-Learning.git

### SETTING UP THE ENVIRONMENT 

The following steps lead to setup the working environment for [CS7641 - Machine Learning](https://www.omscs.gatech.edu/cs-7641-machine-learning) 
in the [OMSCS](http://www.omscs.gatech.edu) program. 

Set up the environment using poetry package manager. This assumes that python is already installed on the local machine

1. Start by downloading the git folder on the local machine.

2. Next, install poetry for your operating system following the instructions here: (https://python-poetry.org/docs/).

3. Once poetry has been install, initialize poetry environment by navigating to the git project folder at the poetry.lock file level and running 'poetry init' on either terminal or command prompt. More information here: https://python-poetry.org/docs/cli/

4. Now that poetry sees the folder and all its sub directories as a poetry module, run 'poetry install' to install all the packages listed in pyproject.toml and poetry.lock file. Those files indicate to the user and poetry respectively on what files are needed in the package.

5. Once poetry has install all packages in the same directory, navigate to the Markov_Decision_Process folder and run 'poetry run python run_experiments.py' to run all the machine learning models
