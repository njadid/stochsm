#!/bin/bash
for i in {1..12}
    do
        echo "Calculating run set $i"
        qsub /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/postprocess/visualize_condsm/template_job.sh -N GPMvis_$i -o /storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/postprocess/visualize_condsm/log_v1/GPMvis_$i.log -F $i
    done