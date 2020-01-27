# Evolutionary-Strategy-Algorithms-From-Scratch
Implementing 4 Evolutionary-Algorithms for 3 Benchmark Functions (Ackley, Rastrigin, Schwefel) <br />
There are 4 categories of evolutionary algorithms:
  1) Genetic Algorithm
  2) Genetic Programming
  3) Evolutionary Strategy
  4) Evolutionary Programming

Evolution Strategy algorithms are used for real value representations. For this reason It is a very good choice for optimizing an objective function. In this repository 4 different Evolutionary Strategy algorithms are implemented from scratch, just using numpy. These algorithms are tested on 3 benchmark functions that have lots of local optimums. These functions are:

1-Ackley Function <br />
<img src="https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/images/ackleyLatex.png" width="50%"><br />
![Image of Yaktocat](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Ackley%27s_function.pdf/page1-300px-Ackley%27s_function.pdf.jpg)<br />

2-Rastrigin Function <br />
<img src="https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/images/rastriginLatex.png" width="35%"><br />
![Image of Yaktocat](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Rastrigin_function.png/300px-Rastrigin_function.png)

3-Schwefel Function <br />
<img src="https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/images/schwefelLatex.png" width="45%"><br />
<img src="https://jamesmccaffrey.files.wordpress.com/2011/12/schwefelsfunction.jpg?w=600&h=476" width="35%">

since These functions have lots of local optimum, it is important to see how do the algorithms work on these functions. Here we implemented 4 different evolutionary strategy algorithms:
  1) self-adaptive 
  2) adaptive with 1/5-success-rule 
  3) PSO 
  4) Differential Evolutionay (DE)
  
