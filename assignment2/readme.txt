COL774 Assignment2

For installing Dependencies run:
pip3 install argparse numpy nltk sklearn seaborn matplotlib cvxopt libsvm

To run executable for question 1:
./run.sh 1 <train_json_file> <test_json_file> <part>

Note: part must be in a-e or g

To run executable for question 2:
./run.sh 2 <train_csv_file> <test_csv_file> <binary_or_multi> <part>

Note:
binary_or_multi can take values 0, 1 for binary, multiclass respectively
part can be a-c for binary, or a-d from multi
