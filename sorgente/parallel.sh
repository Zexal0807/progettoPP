#!/bin/bash

DIR=`dirname $0`

nvcc -w "$DIR"/parallel.cu -o parallel.exe

a=$(date +%s%N)
./parallel.exe
b=$(date +%s%N)
echo $(($b-$a));