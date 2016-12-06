################ 1.7 Second order loss minimization ################

1. Brief Description
2. Instructions for setup
3. Running the project


## 1. Brief Description
Runs loss optimization for Hinge Loss and Cross Entropy Loss with Vanilla, AdaGrad, RmsProp and Adam

## 2. Instructions for setup for development

1. Download and install python from python.org
    Version: minimum python version required 3.5
    URL: https://www.python.org/downloads/


2. API's required
   a. numpy https://docs.scipy.org/doc/numpy/user/install.html
   b. matplotlib http://matplotlib.org/faq/installing_faq.html
   c. sklearn http://scikit-learn.org/stable/install.html
   
## 3. Running the project
To run the complete project with the hardcoded parameters of step=0.001 it takes ~5 mins
Complete code run
python run.py -a

usage: run.py [-h] [-a] [-r] [-p] [-d]

Generate Review Tags

optional arguments:
  -h, --help            show this help message and exit
  -a, --all             Run complete code
  -r, --use-regularizer
                        Use regularizer
  -p, --plot-loss       plot loss for every optimization
  -d, --plot-decision-boundary
                        plot descision boundary for every optimization

For plotting you need close the plot to continue the next optimization



* Train Data : data/train_csv.csv
* Test Data : data/test_csv.csv