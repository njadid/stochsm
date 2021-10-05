## PBS -N ST4_$1          # job name
#PBS -A GT-rbras6-CODA20               # account to which job is charged, ex: GT-gburdell3
#PBS -l nodes=1:ppn=1           # number of nodes and cores per node required
#PBS -l mem=6gb                # memory per core
#PBS -l walltime=200:00:00      # duration of the job (ex: 15 min)
#PBS -j oe                      # combine output and error messages into 1 file
##PBS -o ST4_().out      # output file name
##PBS -m e                    # event notification, set to email on start, end, or fail
##PBS -M navidj@gatech.edu       # email to send notifications to

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

# cd /storage/coda1/p-rbras6/0/njadidoleslam3/download/preprocess_scripts/gpm/
source /storage/coda1/p-rbras6/0/njadidoleslam3/venvs/envdownload/bin/activate

/bin/echo year $1
python /storage/home/hcoda1/6/njadidoleslam3/p-rbras6-0/projects/stochsm/st4_events_yearly.py $1