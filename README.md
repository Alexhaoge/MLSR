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
## Building the Desktop Demo
We write a GUI application as a demo of our model, which is in the `demo` folder. If you want to build it, first ensure `PyQt5` and `pyinstaller` is installed(version specification can be found in `/config/requirement.txt`).
Then type this command in the `demo` folder:
```shell script
pyinstaller -F -w -i favicon.ico MLSR_Demo.py
```
## Run the web demo
The web demo use Flask as backend and a simple html as frontend which can be run on production environment with [Gunicorn](https://flask.palletsprojects.com/en/2.0.x/deploying/wsgi-standalone/).
1. Install the entire repo with all the dependencies and gunicorn.
2. Enter command
```shell
cd /xxxxxx/MLSR/web
gunicorn -b 127.0.0.1:5000 app:app
```
3. It should be running on http://127.0.0.1:5000 and make a reverse proxy by Nginx or Apache if you want it open to the Internet.
