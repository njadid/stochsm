#!/bin/bash
for i in {2000..2020}
    do
        echo "Submitting job for year $i"
        qsub /storage/home/hcoda1/6/njadidoleslam3/p-rbras6-0/projects/stochsm/qsub_gpm_event_extract_yearly.sh -N GPM_$i -o /storage/home/hcoda1/6/njadidoleslam3/p-rbras6-0/projects/stochsm/logs/GPM_$i.log -F $i
    done