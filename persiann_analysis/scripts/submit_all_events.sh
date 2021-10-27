#!/bin/bash
for i in {2000..2020}
    do
        echo "Submitting job for year $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/scripts/qsub_persiann_mbased.sh -N PERS_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/scripts/logs/PERS_$i.log -F $i
    done