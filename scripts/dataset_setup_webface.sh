#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data
if [ ! -d "train" ]; then
  mkdir train
fi

cd train
wget https://owncloud.tuebingen.mpg.de/index.php/s/P5iAosWRFcjLoFf/download -O webface.tar
tar xvf webface.tar
rm webface.tar
