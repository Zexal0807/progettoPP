#!/bin/bash

DIR=`dirname $0`

nvcc -w -lineinfo "$DIR"/parallel.cu -o parallel.exe
