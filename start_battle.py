from generate_data_trath import create_blob
from hinge_loss import hinge_run

if __name__ == '__main__':
    create_blob(plot_fig=False)
    time_taken, acc = hinge_run(plot_fig=False, step = 0.0001, second_ord="vanilla")
    time_taken, acc = hinge_run(plot_fig=False, step = 0.01, second_ord="adam")
