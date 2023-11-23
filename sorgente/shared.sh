#!/bin/bash

DIR=`dirname $0`

nvcc -w "$DIR"/shared.cu -o shared.exe

a=$(date +%s%N)
./shared.exe
b=$(date +%s%N)
echo $(($b-$a));