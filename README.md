# Diagonal-15-Puzzle
Diagonal 15-Puzzle is a modified version of the 15-Puzzle in which the blank square may slide diagonally in addition to horizontal and vertical sliding. However, the cost of each diagonal slide is 1.5 whereas the cost of each horizontal or vertical slide is 1. The final state of the Diagonal 15-Puzzle is as follows:

![Goal State](https://cloud.githubusercontent.com/assets/9055746/22848655/a29af8fc-effe-11e6-824e-f19c51cc42de.JPG)

## Usage
`prj1.py -f (filename) -s (search method {ucs, ils, astar}) -e (heuristic function {1, 2})`  

## Search Methods
1. ucs - Uniform-Cost Search
2. ils - Iterative Lengthening Search
3. astar - A* Search

## Heuristic functions
1. Misplaced tiles
2. My heuristic function

## Additional Notes
1. The command-line argument option `-e` is valid only when the search method `-s` is selected as `astar`
2. If you put your own problem in a text file, the folder where the project is located, you can solve it by entering the name of your file at the command line

## The Solution of the Example Puzzle
![astar h2](https://cloud.githubusercontent.com/assets/9055746/22849025/da1a43ca-f001-11e6-8224-61b8b24f5188.JPG)
