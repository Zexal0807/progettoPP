#!/bin/bash

a=$(date +%s%N)
./shared.exe
b=$(date +%s%N)
echo $(($b-$a));