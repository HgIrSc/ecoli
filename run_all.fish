#!/usr/bin/env fish

set -l data_set
set -al data_set mw1e6_peo1.7_mb20_260127_sample0_top
set -al data_set mw1e6_peo1.7_mb20_260127_sample1_bulk
set -al data_set mw1e6_peo1.0_mb20_260127_sample2_bulk
set -al data_set mw1e6_peo1.0_mb20_260127_sample3_top
set -al data_set mw4e5_peo2.7_mb20_260127_sample5_bulk
set -al data_set mw4e5_peo2.7_mb20_260127_sample4_top

for data in $data_set
    test -d /media/yihang-geng/ExpDataGYH/Ecoli/{$data}_raw
end

if test $status -eq 0
    echo -e '\x1b[32;1mStart processing\x1b[0m'
    for data in $data_set
        echo -e "\\x1b[32;1mProcessing $data\\x1b[0m"
        uv run sub_back.py $data
        uv run traj_extract.py $data
        uv run traj_filter.py $data
        uv run traj_analyze.py $data
    end
end
