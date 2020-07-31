#!/bin/bash

for i in {1..4}

do
   for j in {0.0,0.1,0.2,0.3,0.4,0.5}
	   do
		   python main_2.py --noise_rate=$j
	   done

done



