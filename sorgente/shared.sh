#!/bin/bash

DIR=`dirname $0`

nvcc -w "$DIR"/shared.cu -o shared.exe
