#!/bin/sh
DIR="$HOME/Downloads/SAKINO1024/cbm_rc2"
Tmp=$DIR/tmp.log 

rm $Tmp
ls test*.py >> $Tmp


while read fname
do
    python3 $fname

done <${Tmp}
cat $Tmp
rm $Tmp