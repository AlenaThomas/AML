#!/bin/bash

python test.py

if [ $? -eq 0 ]; then
    echo "Tests passed and commit allowed"
else
    echo "Tests failed and commit aborted"
    exit 1
fi