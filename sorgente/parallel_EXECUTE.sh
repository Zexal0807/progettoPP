#!/bin/bash

a=$(date +%s%N)
./parallel.exe
b=$(date +%s%N)
echo $(($b-$a));