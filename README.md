# Gradient Descent from Scratch

A Python project that implements gradient descent without using any machine learning libraries. It finds the best fit line for a dataset by adjusting slope and intercept values repeatedly until the error is as small as possible.

## What it does

Loads a dataset of x and y values, runs gradient descent to find the best straight line through the data, and saves two charts showing how the error reduced over time and what the final line looks like against the actual data points.

## The tricky part

The learning rate matters a lot here. Too high and the loss doesn't converge, it actually explodes instead of shrinking. I had to tune it down until the descent stabilized and the loss curve actually flattened out instead of blowing up.

## Project structure

gradient-descent/
    data/
        points.csv
    output/
        convergence.png
        line_fit.png
    src/
        model.py
        visualize.py
    main.py
    requirements.txt
    README.md
    .gitignore

## How to run

git clone https://github.com/olarewajumary/gradient-descent
cd gradient-descent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py

## Output

Two charts are saved to the output folder after running:
convergence.png shows the loss dropping over each epoch
line_fit.png shows the final line fitted through the data points

## Libraries used

numpy, pandas, matplotlib