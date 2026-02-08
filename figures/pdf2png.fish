#!/usr/bin/env fish

set -l pdffiles

for file in pdf/*.pdf
    echo found pdf: $file
    set -a pdffiles $file
end

for file in $pdffiles
    set -l pngfile "png/$(path basename $file | path change-extension '')"
    if test -e "$pngfile".png
        echo already exists: "$pngfile".png
    else
        pdftoppm -png -singlefile $file $pngfile
        echo write to png: "$pngfile".png
    end
end
