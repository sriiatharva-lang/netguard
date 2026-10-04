#!/bin/sh
mkdir -p data && cd data
curl -L -o "KDDTrain+.txt" "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
curl -L -o "KDDTest+.txt"  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
