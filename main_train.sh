#!/bin/bash

for i in {1,2,3,4,5}

do
   for j in {0.1,0.2,0.3,0.4}
	   do
		   python new_main_SST.py --noise_rate=$j
	   done

done



