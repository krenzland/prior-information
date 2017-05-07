#!/usr/bin/env bash

./bayes_search.py with level=4 T=0.0 dataset='friedman1' num=20 num_init=5
./bayes_search.py with level=4 T=-0.5 dataset='friedman1' num=20 num_init=5
./grid_search.py with level=5 T=0.0 dataset='power_plant' num=50
