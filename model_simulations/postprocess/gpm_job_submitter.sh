#!/bin/bash
for i in {0..19}
    do
        echo "Calculating run set $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/postprocess/template_job.sh -N GPM_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/postprocess/unify_log_v0/GPM_$i.log -F $i
    done