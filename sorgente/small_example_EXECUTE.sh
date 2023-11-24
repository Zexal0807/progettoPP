#!/bin/bash

a=$(date +%s%N)
./small_example.exe
b=$(date +%s%N)
echo $(($b-$a));