
# Robust Identification of Investor Beliefs
This repository contains codes and a jupyter notebook which estimates and demonstrates results of the empirical example in "Robust Identification of Investor Beliefs" by [Xiaohong Chen][id1], [Lars Peter Hansen][id2] and [Peter G. Hansen][id3]. Latest version could be found [here][id4].

[id1]: https://economics.yale.edu/people/faculty/xiaohong-chen
[id2]: https://larspeterhansen.org/
[id3]: https://mitsloan.mit.edu/phd/students/peter-hansen
[id4]: https://larspeterhansen.org/research/papers/

## Acessing our jupyter notebook
There are two options to access our jupyter notebook. The easiest way is to open a copy in Google Colab by clicking the button below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lphansen/Beliefs/blob/master/Belief_Notebook.ipynb)

Then click "Run all" from "runtime" to see our results. If you are running the notebook the first time, you will need to click the authorization link under the first code cell and copy paste a pop-up string to the input box under the link. 

An alternative way is to store the notebook as well as codes in your local machine. You can do this by following steps below:
1.	Open a Windows command prompt or Mac terminal and change into the folder you would like to store the files. 
    - You can do this using the command ``cd`` in the command prompt (on Windows) or terminal (on Mac).    
    - For example, running ```cd 'C:\Users\username\python'``` (don’t forget '' around the path name to use an absolute path) would lead me to my designated folder.
```
cd [folder path name]
```
2.	Clone the github repository for the paper
    - If you don’t have github installed, try installing it from this page: https://git-scm.com/download.
    - You can do this by running below in the command prompt:
```
git clone https://github.com/lphansen/Beliefs
```
3.	Change directories into the ‘Beliefs’ folder and open jupyter notebook by running below in command prompt:
    - If you don’t have anaconda3 and jupyter notebook installed, try installing from this page: https://jupyter.org/install.html
```
cd Beliefs
jupyter notebook
```
4.	Open ```Belief_Notebook.ipynb```.
5.  Run notebooks cell by cell or click "Run all" from "kernel" in the menu bar to see details about our model results and computational details.   


