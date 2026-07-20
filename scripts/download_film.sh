#!/bin/bash

# Download and re-encode some films.
# Usage: download_film.sh out_dir/

set -x
set -e

VCODEC=libx265
OUT_DIR=$1
mkdir -p $OUT_DIR

# process URL output_name download_section
function process() {
    # Download to tmp.
    yt-dlp $1 -o /tmp/$2.mkv --download-sections $3 -S 'height:1080' 1>/dev/null 2>/dev/null &
    # Waits for current download and previous ffmpeg.
    wait
    # Re encode.
    ffmpeg -y -i /tmp/$2.mkv -ss 15 -c:v $VCODEC -an -crf 30 -r 30 $OUT_DIR/$2.mp4 1>/dev/null 2>/dev/null &
}

# Boom vs Northwestern, Regionals 2026
process 'https://www.youtube.com/watch?v=QgvUhs4TOo4' NW '*8:00-28:00'

# Boom vs Bagnum, Regionals 2026
process 'https://www.youtube.com/watch?v=tiwaaKQtUms' Bagnum '*4:00-24:00'

# Boom vs LaCrosse, MWTD 2026
process 'https://www.youtube.com/watch?v=AuCHyqO3MW8' LaCrosse '*6:00-26:00'

wait
