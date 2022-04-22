#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data
if [ ! -d "train" ]; then
  mkdir train
fi

cd train
wget https://owncloud.tuebingen.mpg.de/index.php/s/8w42X2Kdwwk87fQ/download -O ms1m_refine.tar
tar xvf ms1m_refine.tar
rm ms1m_refine.tar
wget https://owncloud.tuebingen.mpg.de/index.php/s/SddrJjTXWKDQ7Xq/download -O ms1m_refine_train_ann.txt


