from generate_data_trath import create_blob
from hinge_loss import hinge_run

if __name__ == '__main__':
    # Creates and saves the data in data folder
    create_blob(plot_fig=False)
    #Runs hinge loss for the second order plots
    time_taken, acc = hinge_run(plot_fig=False, step = 0.001, second_ord="vanilla")
    time_taken, acc = hinge_run(plot_fig=False, step = 0.01, second_ord="adagrad")
    time_taken, acc = hinge_run(plot_fig=False, step = 0.001, second_ord="rmsprop")
