# Performance-Driven Global Routing
This github repository contains the algorithm and output files for my performance-driven global routing tree optimization (balancing longest path distance and maximum path weight simutaneously). The algorithm used to generate the output files is contained in pa2_pt3.py. It was ran under Python version 3.8, and the packages used were sys and heapq. The 3 output files are:

outputrandom
outputdonut
outputzigzag

These three files contain the optimized trees with the drastically improved Total Weight Ratio (TWR) with respect to a MST and Maximum Distance Ratio (MDR) with respect to a SPT. The code contains extensive documentation on each of its components. The code can be executed by opening the file from IDLE and then running it. The code will print 21 lines, the format of the print is given below.

random graph TWR

random graph MDR

random graph c used

random graph maximum node degree

random graph average node degree

random graph maximum node depth

random graph average node depth

donut graph TWR

donut graph MDR

donut graph c used

donut graph maximum node degree

donut graph average node degree

donut graph maximum node depth

donut graph average node depth

zigzag graph TWR

zigzag graph MDR

zigzag graph c used

zigzag graph maximum node degree

zigzag graph average node degree

zigzag graph maximum node depth

zigzag graph average node depth

The output files will be created in the same folder as the code. Notice that input files must be present in the same folder as the code for the code to run. Otherwise, it wouldn't be able to find input files
