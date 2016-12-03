from generate_data_trath import create_blob
from hinge_loss import hinge_run
from softmax_classifier import cross_entropy_run

if __name__ == '__main__':
    # Creates and saves the data in data folder
    #create_blob(samples=1000, plot_fig=True)
    #Runs hinge loss for the second order plots
    steps = [0.02, 0.01, 0.001, 0.005, 0.0001]
    step = 0.01
    #time_taken, acc = hinge_run(plot_fig=True, step = step, second_ord="vanilla")
    #time_taken, acc = hinge_run(plot_fig=True, step = step, second_ord="adagrad")
    #time_taken, acc = hinge_run(plot_fig=True, step = step, second_ord="rmsprop")
    #time_taken, acc = hinge_run(plot_fig=True, step = step, second_ord="adam")
    cross_entropy_run(consider_reg=False, plot_fig=True)

