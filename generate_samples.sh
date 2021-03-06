#!/usr/bin/bash

nproc=2					# number of processes
nsamp=8000				# number of samples per process
filename=ehs_training.dat	# output filename

echo "Create $nproc processes and generate $nsamp samples each."

for ((i = 1; i < $nproc; i++))
do
	./poker $nsamp $filename.$i > /dev/null &
	echo "Process #$i started"
	sleep 1.1
done

echo "Process #$nproc started"
./poker $nsamp $filename.$nproc  

echo "Script successfully executed."
