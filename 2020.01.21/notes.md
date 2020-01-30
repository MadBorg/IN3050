# Search optimazation

A search problem (optimation):

    We have a basic idea of the shape, and we have a function based on the feeld. say a wing, we have have a wing that gives a flight distance D(w,h) where w is witdh and h is lenght. Then we can make a two dimentional feald to search for best point, And we want to do this in the most efficient way.

* we need a numerical representation
    * (w,h) -> (7,7)
* a function f(x) that tells us how good solution x is.
* A way of finding.
    * 

Two methods

* Continius
  * A continus function
* Descreate
  * Chip design
    * Time tabling
      * Avoid conflicts

## Example:

The salesman problem (TSP)

    Given the coordinates of n cities, find the shortest closed loop.

## Some optimizition

1. Exhaustive
2. Greedy
3. Hill climbing
...

## Exhaustive search (Brute force)

* Test all possible sulutions
* quite expencive
* Discreate?

## Can we be better: Greedy Search

* Only creates and evaluates a single solution
* Makes several locally optimal choices, hoping the res will be near a global optimimum
* Details depend on the implementation.

* Chooses the closest point at every point.

## Hill climber

* compare nighbor solution
  * If a neighbor is better replace the current best
  * Repeat until we reach a certain number of evals.
* Problem: how do we pick a neighbor.
  * Try to swap neighbour until we dont find a better sul.
  

