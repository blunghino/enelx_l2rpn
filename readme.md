# This Starting kit 
This starting kit contains three python notebooks to help you understand and run properly the pypownet platform on chronics to train and test your agents:

- PowerGrid_101_Notebook explains you the problem of power grid operations on a small grid using pypownet platform

- visualize_14grid_with_customized_actions let's you explore the configurations in the action space and visualize it

- RunandSubmitAgent shows you how to define an agent, run them and make it a submission. In particular, it tells you how to check that your submission is valid to run on the competition servers.

You can find the following data in addition:

 - public_data/4_substations : contains one example of a simple 4 subastion gtid for one of the notebook
 - public_data/datasets contains a first set of scenarios : 50 scenarios of one month each
 - public_data/chronic_names.json is used to test locally your submission
 - public_data/full_chronic_names.json contains information to test on all chroncis
theses two files contains indications on the difficulty of each scenario. (0 : easy,1 : medium, 2 : hard)
 - public_data/reward_signal.py : for training purposes you can modify this function, but on codalab we use the default one.
 
 Finally, you will see:
 - output/ which contains some log files you can use them for analysis machine_log.json contains most of the informations you could need: rewards, actions taken, observations.

 - utils/ : this contains files to test your submission in conditions as close as possible as codalab. But you don't need to explore this files.
 - program/ is a link to example_submission/ for testing your submission.
 - example_submission/ where you can code your submission with submission.py.






# with docker 
We have setup a docker container 
```console 
docker run --name l2rpn -it -p 80:8888 -v "$(pwd)":/home/aux marvinler/pypownet:2.2.7 jupyter-notebook --ip 0.0.0.0 --allow-root --notebook-dir=/home/aux --NotebookApp.token=""
```


For a light version of the docker (withous (deep)-learning libraries)
```console 
marvinler/pypownet:2.2.7-light
```

Then open localhost on your web browser


# without Docker
Install pypownet
```console 
git clone https://github.com/MarvinLer/pypownet
cd pypownet
pip install . --user
```
for more informations you can read the documentation of pypownet 
then you can run pypownet locally.
```console
python -m pypownet.main -p parameters/default14 -lv level0 -a DoNothing 
```

You can now run the notebooks in this directory.
```console
jupyter-notebook
```
pypownet has gym compatibility for RL training.







for morte information about this challenge please visit the challenge webpage
https://l2rpn.chalearn.org/

and you can look an web game on managing a small power network
https://credc.mste.illinois.edu/applet/pg


