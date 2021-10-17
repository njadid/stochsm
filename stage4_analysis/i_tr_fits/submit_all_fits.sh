#!/bin/bash
for i in {1..12}
    do
        echo "Submitting job for year $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/i_tr_fits/qsub_fits.sh -N FIT_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/i_tr_fits/logs/FIT_$i.log -F $i
    done
