#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for month $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/qsub_summary.sh -N SUMST4_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/log/summaryST4_$i.log -F $i
    done