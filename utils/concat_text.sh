#!/bin/bash

find $1 -type f -name '*.txt' -exec cat {} + > test.txt