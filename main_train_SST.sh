#!/bin/bash

for i in {1,2,3}

do
   for j in {0.45,0.47,0.5}
	   do
		   python main_SST.py --noise_rate=$j
	   done

done



