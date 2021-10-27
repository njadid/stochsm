#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for month $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/scripts/visualizations/qsub_vis.sh -N VIS_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/scripts/visualizations/logs/VIS_$i.log -F $i
    done
