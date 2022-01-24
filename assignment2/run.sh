q=$1;
train_file=$2;
test_file=$3;

if [[ ${q} == "1" ]]; then
python3 nb.py $2 $3 $4
fi

if [[ ${q} == "2" ]]; then
python3 svm.py $2 $3 $4 $5
fi

