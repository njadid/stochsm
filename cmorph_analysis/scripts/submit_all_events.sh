#!/bin/bash
for i in {2001..2020}
    do
        echo "Submitting job for year $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/gpm_analysis/scripts/qsub_gpm_mbased.sh -N GPM_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/gpm_analysis/scripts/logs/GPM_$i.log -F $i
    done