#!/bin/bash
for i in {2000..2020}
    do
        echo "Submitting job for year $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/qsub_st4_new.sh -N ST4_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/logs/ST4_$i.log -F $i
    done