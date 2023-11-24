#!/bin/bash
a=$(date +%s%N)
./complex_example.exe
b=$(date +%s%N)
echo $(($b-$a));