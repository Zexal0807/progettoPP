#!/bin/bash

DIR=`dirname $0`

nvcc -w -lineinfo "$DIR"/shared.cu -o shared.exe
