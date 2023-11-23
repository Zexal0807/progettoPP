#!/bin/bash

DIR=`dirname $0`

g++ "$DIR"/small_example.cpp -o small_example.exe

a=$(date +%s%N)
./small_example.exe
b=$(date +%s%N)
echo $(($b-$a));