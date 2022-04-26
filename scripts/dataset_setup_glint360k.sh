#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data
if [ ! -d "train" ]; then
  mkdir train
fi

cd train
wget https://keeper.mpdl.mpg.de/f/689ebd19842b476280e3/?dl=1 -O glint360k.tar
tar xvf glint360k.tar
rm glint360k.tar
