#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for month $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/qsub_summary.sh -N ST4_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/logs/summaryST4_$i.log -F $i
    done