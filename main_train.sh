#!/bin/bash

for i in {1,2,3,4,5}

do
   for j in {0.6,0.7}
	   do
		   python main_SST.py --noise_rate=$j
	   done

done



