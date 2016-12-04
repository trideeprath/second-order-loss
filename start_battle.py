from generate_data_trath import create_blob
from hinge_loss import hinge_run
from softmax_classifier import cross_entropy_run

if __name__ == '__main__':
    plot_status = False
    # Creates and saves the data in data folder
    #create_blob(samples=1000, plot_fig=plot_status)
    #Runs hinge loss for the second order plots
    plot_status = False
    steps = [0.02, 0.01, 0.001, 0.005, 0.0001]
    step = 0.01
    consider_regs = [True, False]
    for reg in consider_regs:
        time_taken, acc, iteration_count = hinge_run(plot_fig=plot_status, step = step, second_ord="vanilla", consider_reg=reg)
        time_taken, acc, iteration_count = hinge_run(plot_fig=plot_status, step = step, second_ord="adagrad", consider_reg=reg)
        time_taken, acc, iteration_count = hinge_run(plot_fig=plot_status, step = step, second_ord="rmsprop", consider_reg=reg)
        time_taken, acc, iteration_count = hinge_run(plot_fig=plot_status, step = step, second_ord="adam", consider_reg=reg)
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, second_ord="vanilla")
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, second_ord="adagrad")
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, second_ord="rmsprop")
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, second_ord="adam")
