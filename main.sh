#!/bin/bash

for i in {0.0001,0.0005,0.001,0.0015,0.002}

do
   for j in {1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5}
	   do
		   python main_2.py --epsilon=$j --lr=$i
	   done

done



