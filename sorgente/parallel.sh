#!/bin/bash

DIR=`dirname $0`

nvcc -w "$DIR"/parallel.cu -o parallel.exe
