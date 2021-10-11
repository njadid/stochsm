#!/bin/bash
for i in {2000..2020}
    do
        echo "Submitting job for year $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/qsub_st4_mbased.sh -N ST4_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/log/ST4_$i.log -F $i
    done