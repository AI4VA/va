## Preprocessing
The scripts are to pre-process the Joern-generated nodes.csv and edges.csv into GGNN inputs.

Usage: 
* python3 normalize.py [repo1]
* python3 csv2gnninput --src [repo2] --csv [repo3]

Note: 

repo1 is Joern-generated repository, the repository contains several

sub-repositories, and each is corresponding to a .c file

repo2 is the repository containing the original .c files

repo3 is the normalized version of repo1


Please see the example_repo1, example_repo2, example_repo3 for details.

