#!/bin/bash


files=( $( find snapshot -type f -newermt '8/9/2020  19:50:00'  ) )

prev='asdf'
for filename in ${files[@]} ; do
       snapshot=`echo "$filename" | cut -d "/" -f2`
       if [ $prev != $snapshot ]; then
	  snap2=`ls snapshot/$snapshot -Art | tail -n 1`
	  snap=`ls $filename -Art | cut -d "/" -f2`
	  python main_DBpedia_infer.py --snapshot=snapshot/$snap/$snap2
       fi
       prev=$snapshot
done
