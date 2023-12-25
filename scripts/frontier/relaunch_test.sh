#!/bin/bash

jobid=$1

function isolate () {
    sacct -j $jobid -B | grep "^$1" | sed "s/$1//"
}

state=$(sacct -j $jobid --format=JobID,state | grep "$jobid " | sed "s/[0-9]* //g")
while [[ "$state" != "COMPLETE" ]]
do
  while [[ $(squeue --user alexisroger | grep $jobid) ]]
  do
	echo -n .
	sleep 60
  done


  path=$(isolate 'CHECKPOINT_PATH=')
  name=$(isolate '    --output_dir $CHECKPOINT_PATH' | sed "s/ //g" | sed 's/\\//g')
  eval "edir=$path$name/"

  cp /lustre/orion/csc538/scratch/$(whoami)/job_logs/*-$jobid.* $edir


  #sacct -j $jobid -B | tail -n +3 | sbatch
  tmp=$(sacct -j $jobid -B | tail -n +3 | sbatch)
  jobid=$(echo "$tmp" | grep -oG "[0-9]*")
  echo "Launching job $jobid"
  state=$(sacct -j $jobid --format=JobID,state | grep "$jobid " | sed "s/[0-9]* //g")
done

echo 'DONE!!!'
