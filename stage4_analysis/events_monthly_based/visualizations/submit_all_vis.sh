#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for year $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/visualizations/qsub_vis.sh -N VIS_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/visualizations/logs/VIS_$i.log -F $i
    done

# qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/visualizations/qsub_vis.sh -N VIS_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_monthly_based/visualizations/logs/VIS_$i.log -F $i