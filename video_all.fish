#!/usr/bin/env fish

# set -l targets
set -l targets $(find /media/yihang-geng/ExpDataGYH/Ecoli/ -maxdepth 1 -type d -name '*2601*')
# set -al targets mw1e6_peo1.7_mb20_260127_sample0_top
# set -al targets mw1e6_peo1.7_mb20_260127_sample1_bulk
# set -al targets mw1e6_peo1.0_mb20_260127_sample2_bulk
# set -al targets mw1e6_peo1.0_mb20_260127_sample3_top
# set -al targets mw4e5_peo2.7_mb20_260127_sample5_bulk
# set -al targets mw4e5_peo2.7_mb20_260127_sample4_top

for target in $targets
    # set -l tifdir /media/yihang-geng/ExpDataGYH/Ecoli/{$target}_raw
    set -l tifdir $target
    echo find dir: $tifdir
    set -l basename $(path basename $tifdir)
    set -l video_path1 /media/yihang-geng/ExpDataGYH/Ecoli/videos/{$basename}.avi
    set -l video_path /home/yihang-geng/Workspace/EcoliViscotaxis/videos/{$basename}.avi
    if test -e $video_path1
        set_color green
        echo already exists: $video_path
        set_color normal
    else
        set_color blue
        echo save to: $video_path
        set_color normal
        ffmpeg -framerate 20 \
            -pattern_type glob -i "$tifdir"/t\*0.tif \
            -vcodec rawvideo -pix_fmt gray $video_path
    end
end
