Documentation:
This code is for COL774 assignment 1. 
Code and plots for each question are in a separate folder labelled q1, q2, q3, q4.
To view help/usage for each file, run 'python3 *.py -h'

Q1:
    2 input files(for X and y) are required. To run code:
    python3 lin_reg.py <path_to_file_X> <path_to_file_y> [<flags>]

    See optional flags below:
        usage: lin_reg.py [-h] [--l_rate L_RATE] [--error ERROR]
                  [--animation_framesize ANIMATION_FRAMESIZE]
                  fileX fileY

    Implementation of batch linear regression using gradient descent

    positional arguments:
    fileX                 csv file for X input
    fileY                 csv file for Y input

    optional arguments:
    -h, --help            show this help message and exit
    --l_rate L_RATE, -l L_RATE
                            Learning Rate for gradient Descent
    --error ERROR, -e ERROR
                            threshold for min_error
    --animation_framesize ANIMATION_FRAMESIZE, -f ANIMATION_FRAMESIZE
                            iterations covered in one frame of animation
    If optional flags are not set, values of corresponding will be taken from the code

Q2:
    1 input test file is required parameter. To run code:
    python3 sgd.py <path_to_file> [<flags>]

    See optional flags below:
        usage: sgd.py [-h] [--l_rate L_RATE] [--batch_size BATCH_SIZE]
                [--n_check_conv N_CHECK_CONV] [--error ERROR]
                [--plot_iter_to_skip PLOT_ITER_TO_SKIP]
                file_test

        implementation of SGD

        positional arguments:
        file_test             csv test file

        optional arguments:
        -h, --help            show this help message and exit
        --l_rate L_RATE, -l L_RATE
                                Learning Rate for SGD
        --batch_size BATCH_SIZE, -b BATCH_SIZE
                                Batch size for SGD
        --n_check_conv N_CHECK_CONV, -n N_CHECK_CONV
                                Checks convergence after every n batches
        --error ERROR, -e ERROR
                                threshold for convergence
        --plot_iter_to_skip PLOT_ITER_TO_SKIP, -s PLOT_ITER_TO_SKIP
                                Plot data for every s iterations

    If any optional flags is not set, values of corresponding variable will be taken from the code

Q3: 
    2 input files are required, for X and y values:
    python3 log_reg.py <path_to_file_X> <path_to_file_y> [<flags>]

    See optional flags below:
        usage: log_reg.py [-h] [--error ERROR] fileX fileY

        Implementation of Logistic Regression using Newton's Method

        positional arguments:
        fileX                 File for input X
        fileY                 File for input Y

        optional arguments:
        -h, --help            show this help message and exit
        --error ERROR, -e ERROR
                                threshold for convergence
    If optional flags are not set, values of corresponding will be taken from the code

Q4: 
    Takes 2 input files, for X and y values:
    python3 log_reg.py <path_to_file_X> <path_to_file_y> [<flags>]

    See usage help below:
        usage: gda.py [-h] fileX fileY

        Implementation for GDA

        positional arguments:
        fileX       File for X input
        fileY       File for Y input

        optional arguments:
        -h, --help  show this help message and exit


For Q1, Q2, Q3, Q4: plots, animations are inside the '/plots' folder


