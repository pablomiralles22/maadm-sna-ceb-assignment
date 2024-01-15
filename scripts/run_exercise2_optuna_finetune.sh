
NUM_PROCS=6 # number of processes to run in parallel for each metric
NUM_TRIALS=20 # NUM_TRIALS * NUM_PROCS = total number of trials
BUDGET_FITNESS_CALLS=5000 # low compute :(
RUNS_PER_EXPERIMENT=1 # low compute :(
METRIC_1="internal_density"
METRIC_2="avg_odf"

pids=""
for i in $(seq 1 $NUM_PROCS)
do
    python scripts/exercise2_optuna_finetune.py         \
        --graph-file data/amazon_graph.graphml          \
        --budget-fitness-calls $BUDGET_FITNESS_CALLS    \
        --num-trials $NUM_TRIALS                        \
        --proc-id $i                                    \
        --metric-1 $METRIC_1                            \
        --metric-2 $METRIC_2                            \
        --runs-per-experiment $RUNS_PER_EXPERIMENT      \
        --study-name "amazon_$METRIC_1-$METRIC_2" &

    pids+=" $!"

    # if first, sleep 4 seconds to allow the first process to create DB
    if [ $i -eq 1 ]
    then
        sleep 5
    fi
done

echo "Run 'kill -9$pids' to stop the process"