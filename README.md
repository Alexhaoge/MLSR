# MLSR
Maching Learning Scholarship Rater(MLSR), is a rating assistant for scholarship that helps rate the applicants based on maching-learning methods.  
**Clarification**: This project currently is incomplete, and **it is only for research, not application**. Some rules (for the convenience of data manipulation) of model such as feature engineering may seems discriminatory **but we are not intended to do so**.
## Install
1. install dependency
If you use `pip`, run
```shell script
pip install -r config/requirement.txt
```
2. load data in `data` folder. Due to privacy protection we hide our original dataset so you need to construct your own dataset .   
`Demo.csv` is an example for dataset format.  
## Run
1. For model training  
```shell script
python main.py --dt
# This means training decision tree.
```
Type `python main.py --help` for command of other models.  
**Note**: you need to create the model output directory before training!  
2. For plot, run `python plot_main.py`. (You may need to modify this code to draw the plot you want.)
## API Documentation
[click here](https://www.alexhaoge.xyz/mlsr/index.html)
