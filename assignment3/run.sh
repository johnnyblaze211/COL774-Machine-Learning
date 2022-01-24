q=$1;

if [[ ${q} == "1" ]]; then
python3 dt.py $2 $3 $4 $5
fi

if [[ ${q} == "2" ]]; then
python3 nn.py $2 $3 $4
fi