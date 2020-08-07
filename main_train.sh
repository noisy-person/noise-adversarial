#!/bin/bash

for i in {1,2,3,4,5,6}

do
   for j in {0.7}
	   do
		   python main_DBpedia.py --noise_rate=$j
	   done

done



