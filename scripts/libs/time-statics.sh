


func_1_8_get_current_milliseconds() {
    date +%s%3N
}

# start timer: $1 = variable name to store start time
func_1_9_start_time_count() {
    local -n _timer_ref=$1
    # _timer_ref=$(date +%s)
    _timer_ref="$(func_1_8_get_current_milliseconds)"
}

# format duration from milliseconds to human readable string
# $1 = milliseconds
# $2 = variable name to store formatted result
func_1_10_format_duration_ms() {
    local ms=$1
    local -n _out_ref=$2

    if (( ms < 1000 )); then
        _out_ref="${ms} ms"
        return
    fi

    if (( ms < 60000 )); then
        # seconds with 2 decimal places
        _out_ref=$(printf "%.2f seconds" "$(awk "BEGIN { print $ms / 1000 }")")
        return
    fi

    # minutes + seconds
    local total_seconds minutes remaining_ms
    total_seconds=$(( ms / 1000 ))
    minutes=$(( total_seconds / 60 ))
    remaining_ms=$(( ms % 60000 ))

    _out_ref=$(printf "%d minutes %.2f seconds" \
        "$minutes" \
        "$(awk "BEGIN { print $remaining_ms / 1000 }")")
}

# finalize timer:
# $1 = start time variable
# $2 = end time variable
# $3 = the variable name to store formatted elapsed time string
func_1_11_elapsed_time_calculation() {
    local start_time=$1
    local end_time=$2
    local -n elapsed_string_ref=$3

    # func_1_1_log "\$1=$1 ; \$2=$2; \$3=$3"

    local elapsed_time=$((end_time - start_time))

    func_1_10_format_duration_ms "${elapsed_time}" elapsed_string_ref
}


func_2_4_finalize() {
    local exit_code=$?
    local final_time=$(func_1_8_get_current_milliseconds)

    if [[ -n "${__START_TIME:-}" ]]; then

        # Processing -- 1. second stage build time statistics
        if [[ ! -z "${__SECOND_STAGE_BUILD_START_TIME:-}" ]]; then
            local second_duration_human
            func_1_11_elapsed_time_calculation "${__SECOND_STAGE_BUILD_START_TIME}" "${final_time}" second_duration_human
            func_1_1_log "Build elapsed time: ${second_duration_human}."
        fi

        # Processing -- 2. total elapsed time statistics
        local whole_duration_human
        func_1_11_elapsed_time_calculation "${__START_TIME}" "${final_time}" whole_duration_human
        func_1_1_log "Total elapsed time: ${whole_duration_human}."
    else
        func_1_1_log "Elapsed time information not available."
    fi

    exit "$exit_code"
}



