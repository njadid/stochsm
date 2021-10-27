#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for month $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/scripts/qsub_persiann_monthly_agg.sh -N PERS_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/scripts/logs/MonthlyPERS_$i.log -F $i
    done