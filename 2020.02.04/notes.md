# Population managment, ...

OBLIG IS PUBLISHED

## Selection Chapter 5

* Selection: is a second fundamental force for evolutionary algorithm
* We want to select our parents. Some of them good and some of them not soo good. And change them some. All in a stocastick maner.
* The selection operators are representation independant because they work on the fitnes value.
* We only care about the value representing the goodnes of fit.
* High selection pressure, then we only select the strong one, but with lower we explore more.
* Espessialy in the beginning we want to explore more.

## Fitnes selection

### Propartions Selection

* Roulette wheel selection
* Prob of a selection an invidual: P(i) = f_i / Sum(f_j)
* Problem

### Rank based selection

* Atempting to solve the problems with FPS
* Ranks the fitnes to dont loose fitnes pressure?

### Tournament Selection

* Have the players compete.
* Only local fitnes information
* Pick k random sulutions and and find the winners.
* Prob of picking i is dependent on
  * Rank if i
  * Size of sample k
    * Hugher k is higher selection pressure
  * Pick with replacement

## Survivor replacment

mu - poplulation of old sulutions

lambda - offspring

we want to keep the pop stable.

ways to pick: Random, Only the best, Some of the best and some of the rest.

Elitism: So we dont loose the N best solutions, then to have randomnes in the rest.

(mu, lambda) - Selection: Based on the set of children only (lambda > mu)

## Explisit 
### Modularity

* We want the global max
* different peeks might be good to know


### Preserving Diversity:

* Restricting the number if individuals within a given nich by " sharing their fitnes.
* Fitnes function to detetmin if the solutions are similar enoght
* Keeps many peeks. So we will find many peeks.

### Crowding

* One offsprin only kompeeting with the parents
* Res: Even distribution aming niches

## Implicit Approches for preserving Diversity:

* Only mate with similar geno or pheno type
* Add species tags to genotype, so that only similar species can mate.

### Geographical separation

* Run several algorithms independently, but every Nth run we mix some of the ilands.

## Hibridisation with Other Techniques: memetic algorithms

### Why

* Might be looking at improving on existing techniques (non - EA)
* Might be looking at improving EA search for good solutions
* We use some knowlage to solve some of the problem

### What is memetic?

* WE use some knolage about the problem

### Local search: main idea

Make a small, intellegent change, with an existing method.
If change improves it continue.
Otherwise keep trying small.

#### Pivot rules

* Greedy accent
  * We stop as soon as we have found a bette solution
* Steepest accent
  * Trying all the nighbours

### Local Search end evolution

Do offspring inherit what their parents have "learnded" in life?

* Yes - Lamarchian evolution
  * Improved fiteness and genotype
* No - Bakdwinian evolution
  * Improved fitness only

### Where to hybridise

* Start the pop with some knolage. Known decent solutions.
* Problem specific for mating pool: Crossover
* Specific mutation
* Local search

## Multibojective Evolutionary algorithms

Multiple objective solves for multiple objectives. We cant get your vaurite car and a scheep car. But somwhere inbetween.  find a copremize.

### Weighted sum

* Transforme into a single objective

### A set of multi objective solutions (pareto front)

* The pop based nature of EAs ued to simultanueously search for a set of points approcimating pareto front

## Pareto optimality

* Solution x is non-dominated among a set of solutions Q if no solution from Q dominates x
* A set of non-dominated solutions from the entire feasible solution space is the Pareto set, or Pareto front, its members Pareto-optimal solutions 

* Trying to find the pareto line. So all the solutions along the line.

## EC approch

### Selection

Dominance:

* ranking or bepth based
* fitness related to while population

### divercity maintanence

* Even distribution

### Remember Good Points

* culd just use an elitis algorithm
* Common to maintain an archive of non dominated points
  * to store the good ones


