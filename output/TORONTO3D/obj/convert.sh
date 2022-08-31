#!/bin/bash

for file in  ./*.obj
do
    echo 'converting' $file 
    obj23dtiles -i $file --tileset -p customTilesetOptions.json
done
