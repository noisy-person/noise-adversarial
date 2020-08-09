#!/bin/bash

for i in {1,2,3,4,5}

do
   for j in {0.2,0.4}
	   do
		   python main_TREC.py --noise_rate=$j
	   done

done



