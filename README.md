# Gradient Descent from Scratch

A Python project that implements gradient descent without using any machine learning libraries. It finds the best fit line for a dataset by adjusting slope and intercept values repeatedly until the error is as small as possible.

## What it does

Loads a dataset of x and y values, runs gradient descent to find the best straight line through the data, and saves two charts showing how the error reduced over time and what the final line looks like against the actual data points.

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

Install dependencies:
pip install -r requirements.txt

Run the project:
python main.py

## Output

Two charts are saved to the output folder after running:
convergence.png shows the loss dropping over each epoch
line_fit.png shows the final line fitted through the data points

## Libraries used

numpy, pandas, matplotlib