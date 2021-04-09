#! /bin/bash

get_unused_port() {
    # Well-known ports end at 1023.  On Linux, dynamic ports start at 32768
    # (see /proc/sys/net/ipv4/ip_local_port_range).
    local MIN_PORT=1024
    local MAX_PORT=32767

    local USED_PORTS=$(netstat -a -n -t | tail -n +3 | tr -s ' ' | \
        cut -d ' ' -f 4 | sed 's/.*:\([0-9]\+\)$/\1/' | sort -n | uniq)

    # Generate random port numbers within the search range (inclusive) until we
    # find one that isn't in use.
    local RAN_PORT
    while
        RAN_PORT=$(shuf -i 1024-32767 -n 1)
        [[ "$USED_PORTS" =~ $RAN_PORT ]]
    do
        continue
    done

    echo $RAN_PORT
}

#
# Initializes information about the nodes for the current run.
#
init_node_info() {
    export PRIMARY=$(hostname -s)
    SECONDARIES=$(scontrol show hostnames $SLURM_JOB_NODELIST | \
        grep -v $PRIMARY)

    ALL_NODES="$PRIMARY $SECONDARIES"
    export PRIMARY_PORT=$(get_unused_port)
}

#
# Runs a command on a host node and will attempt to re-run the command upon
# failure. The command to run must be passed as an argument; the maximum number
# of tries and timeout before a retry can also be passed as arguments.  The
# default MAX_TRIES is 3 and the default WAIT_TIME is 4.
#
# Usage: run_with_retry "$COMMAND" [$MAX_TRIES [$WAIT_TIME]]
#
run_with_retry() {
    # Get function arguments, using defaults as needed.
    local COMMAND=$1

    local MAX_TRIES=$2
    if [ -z $MAX_TRIES ]
    then
        MAX_TRIES=2
    fi

    local WAIT_TIME=$3
    if [ -z $WAIT_TIME ]
    then
        WAIT_TIME=60
    fi

    local TRY_N=0
    local RETVAL=1

    # Run the command, retrying as needed until MAX_TRIES is exceeded.
    while
        TRY_N=$(($TRY_N + 1))

        $COMMAND
        RETVAL=$?

        [ $MAX_TRIES -gt $TRY_N ] && [ $RETVAL != 0 ]
    do
        echo "Run failed on $(hostname -a); retrying in $WAIT_TIME second(s)..."
        sleep $WAIT_TIME
    done

    if [ $RETVAL != 0 ]
    then
        echo "Run failure on $(hostname -a); maximum tries exceeded."
    fi
}