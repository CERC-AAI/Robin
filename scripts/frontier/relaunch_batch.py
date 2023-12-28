jobids = [
    1552987,
    1553019,
    1552391
]

import os
import subprocess
from time import sleep

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip()

def cp_logs(jobid):
    return run(f"""
function isolate () {{
       sacct -j {jobid} -B | grep "$1" | sed "s/$1//" | sed "s/ //g"
}}

NAME=$(isolate 'NAME=')
CHECKPOINT_PATH=$(isolate 'CHECKPOINT_PATH=')
outdir=$(isolate '\--output_dir' | sed 's/\\\\//g')
eval "outdir=$outdir"
eval "outdir=$outdir"
echo "$outdir"

cp /lustre/orion/csc538/scratch/$(whoami)/job_logs/*-{jobid}.* $outdir/
    """)

while len(jobids) > 0:
    new_jobids = jobids.copy()
    for jobid in jobids:
        status = run(f'sacct -j {jobid} --format=JobID,state | grep "{jobid} " | sed "s/[0-9]* //g"')[0]
        print(status)
        match status:
            case "PENDING":
                pass
            case "RUNNING":
                pass
            case "COMPLETED":
                print(f"Job {jobid} is completed")
                new_jobids.remove(jobid)
            case "FAILED":
                print(f"Job {jobid} has failed")
                new_jobids.remove(jobid)
            case "TIMEOUT":
                cp_logs(jobid)
                new_id = run(f'echo "$(sacct -j {jobid} -B | tail -n +3 | sbatch)" | grep -oG "[0-9]*"')
                print(f"Job {jobid} has timeout, relaunching as {new_id}")
                new_jobids.remove(jobid)
                new_jobids.append(int(new_id))
            case _:
                print(f"Unknown job status: {jobid} {status}")

    jobids = new_jobids
    sleep(60)
                