#! /bin/bash

echo -e "kernel_size\tnpp_time\tcuda_time" > output.txt
for arg in 3 5 7; do
    for file in images/aerials_original/*; do 
        [ -f "$file" ] && echo "Processing: $file";
        ./build/blur_project $arg $file >> output.txt;
    done;
done