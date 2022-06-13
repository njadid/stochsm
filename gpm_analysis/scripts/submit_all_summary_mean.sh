#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for month $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/gpm_analysis/scripts/qsub_summary_mean.sh -N GPM_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/gpm_analysis/scripts/logs/summaryGPMMean_$i.log -F $i
    done