from generate_data_trath import create_blob
from hinge_loss import hinge_run
from softmax_classifier import cross_entropy_run
from argparse import ArgumentParser

def parse_cli_parameters():
    parser = ArgumentParser(description="Generate Review Tags")
    parser.add_argument('-a', '--all', dest='all', action="store_true",
                        help='Run complete code', default=False)
    parser.add_argument('-r', '--use-regularizer', dest='regularizer', action="store_true",
                        help='Use regularizer', default=False)
    parser.add_argument('-p', "--plot-loss", dest='plot_loss', action="store_true",
                        help='plot loss for every optimization', default=False)
    parser.add_argument('-d', "--plot-decision-boundary", dest='plot_decision', action="store_true",
                        help='plot descision boundary for every optimization', default=False)

    options = parser.parse_args()

    return options


if __name__ == '__main__':
    cli_parameters = parse_cli_parameters()
    if cli_parameters.regularizer is True:
        consider_regs = [True]
    else:
        consider_regs = [False]

    if cli_parameters.all is True:
        consider_regs = [True, False]

    if cli_parameters.plot_loss is True:
        print("You have to close the Figures to continue runnning the code")
        plot_iteration = True
    else:
        plot_iteration = False

    if cli_parameters.plot_decision is True:
        print("You have to close the Figures to continue runnning the code")
        plot_status = True
    else:
        plot_status = False

    # Creates and saves the data in data folder
    #Create_blob(samples=1000, plot_fig=plot_status)
    #Runs hinge loss for the second order plots
    step = 0.001
    stop_hinge, stop_cross = 0.001, 0.0001
    for reg in consider_regs:
        hinge_run(plot_fig=plot_status, plot_iteration=plot_iteration, step=step, second_ord="vanilla", consider_reg=reg, stop=stop_hinge)
        hinge_run(plot_fig=plot_status, plot_iteration=plot_iteration, step = step, second_ord="adagrad", consider_reg=reg, stop=stop_hinge)
        hinge_run(plot_fig=plot_status, plot_iteration=plot_iteration, step = step, second_ord="rmsprop", consider_reg=reg, stop=stop_hinge)
        hinge_run(plot_fig=plot_status, plot_iteration=plot_iteration, step = step, second_ord="adam", consider_reg=reg, stop=stop_hinge)
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, plot_iteration =plot_iteration, second_ord="vanilla", stop=stop_cross)
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, plot_iteration =plot_iteration, second_ord="adagrad", stop=stop_cross)
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, plot_iteration =plot_iteration, second_ord="rmsprop", stop=stop_cross)
        cross_entropy_run(consider_reg=reg, plot_fig=plot_status, plot_iteration =plot_iteration, second_ord="adam", stop=stop_cross)
