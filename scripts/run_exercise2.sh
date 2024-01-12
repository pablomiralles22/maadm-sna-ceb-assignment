mkdir -p logs

NUM_RUNS=6
NUM_TRIALS=15

pids=""
for i in $(seq 1 $NUM_RUNS)
do
    python scripts/exercise2.py --graph-file data/amazon_graph.graphml --num-trials $NUM_TRIALS --log-file logs/process$i.txt &
    pids+=" $!"
    # if first, sleep 4 seconds to allow the first process to create DB
    if [ $i -eq 1 ]
    then
        sleep 5
    fi
done

echo "Run 'kill -9$pids' to stop the process"