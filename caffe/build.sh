#!/bin/bash
rm -rf cmake_build
mkdir cmake_build
cd cmake_build
cmake ..
make -j128
make install
