# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <br>

# # Grid Visualization
#
# This notebook is created to visualize simple changes over the IEEE 14 grid and to gain a better understanding on how to execute actions using pypownet. The idea is to present a straightforward notebook where the user could pick any substation id among the grid and play modifying its internal configuration to see the changes.
#
# In order to visualize the grid, the external python script `visualize_grid.py` has to be imported (the starting kit already holds the mentioned code). However, the script relies on other libraries more than those required by pypownet. If you have not installed them yet, please try the following in your terminal.
# <br><br>
# `pip3 install --user networkx plotly`<br>
# `jupyter labextension install @jupyterlab/plotly-extension` if using jupyter lab
# <br>

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from utils.visualize_grid import plot_grid
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

# Connect Plotly in offline mode. 
init_notebook_mode(connected = True) 
# -

# It is important as well to import the pypownet's **environment** and **runner** which allows us to get the respective grid's state with all power flow results.

# +
import os
import pypownet.environment
import pypownet.runner

# Initialize the env.
environment = pypownet.environment.RunEnv(parameters_folder=os.path.abspath('public_data'),
                                          game_level='datasets',
                                          chronic_looping_mode='natural', start_id=0,
                                          game_over_mode='hard' )
# -

# print dir(environment)
action_space = environment.action_space
observation_space = environment.observation_space
game = environment.game

# ## What is an action?
#
# An action is that an intercation an agent can do with its environment. It our specific case, we emulate what operators can do in a control room to control the system. Typically, there are two actions allowed on the grid:
#
# - Change the status of a line (switch ON or OFF tranmission lines)</li>
# - Switch to another busbar the substation's elements, such as productions, consumptions or power lines in the same substation (node-splitting).
#     
# These two types of emulated actions constitute an **Action** in pypownet. The *action vector* is concatenation of two binary lists: the first list corresponds to all elements switches in all ***substations*** and the other for ***transmission lines***. For the sake of simplicity, the following figure is a representation of the action vector for the four substation network you saw in the 101_notebook.
#
# <img src="http://i66.tinypic.com/2vmj682.jpg" width=850 ALIGN="middle">
#
# At this point you may notice some elements inside a substation list are tagged as **or** and **ex**. These elements belong uniquely to the transmission switch associated with one extreme of a line. By convention, all lines should have an **or**igin and an **ex**tremity as we indicate as follows.
#
# <img src="http://i63.tinypic.com/2ccv2ac.png" width=350 ALIGN="middle">

# ### How to interpret the action vector?
#
# The action vector is an array of binary numbers with all action allowed in the grid. By default in pypownet, all elements in the grid are in the state ON or online and connected to a single busbar in a substation. But what do exactly these binary values mean? 
#
# <img src="http://i64.tinypic.com/i2oto7.jpg" width=230 ALIGN="middle">
#
# 1. Action in transmission lines: <br>
# 1.1. A value of 1 in the **line list** means ***change*** the current status of a line (if a line is ON, it will switch it off and viceversa).<br>
# 1.2. A value of 0 in the **line list** means ***do not change*** the current status of a line (if a line is ON, it will remain ON and viceversa).
# 2. Action in substations: <br>
# 2.1. A value of 1 in the **substation list** means ***switch*** the selected element to the other busbar in the same substation (For instance a or transmission is connected to one busbar then it will be switch to the second one in a double busbar configuration). <br>
# 2.2. A value of 0 in the **substation list** means ***do not switch*** the selected element to the other busbar in the same substation (For instance a or transmission will remain connected to the same busbar where it was connected before the action).
#
# ***
# For a better understading, please refer to the official [pypownet documentation](https://pypownet.readthedocs.io/en/latest/env_info.html#action-understanding) and the Introduction to transmission system operation [notebook](Power_Grid_101_notebook.ipynb).
# ***

# Let's now initialize the action class that will be used later to make changes on the grid.

action = action_space.get_do_nothing_action(as_class_Action=True)

# ### Chose a substation id

# +
# # ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ 
# Play with this variable to see the change in topology for other nodes

sub_id = 2

# # ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ 

expected_target_configuration = action_space.get_number_elements_of_substation(sub_id)
target_configuration = np.zeros(expected_target_configuration)
# -

# ### Target configuration
# The target configuration is a binary array whose numbers represent the action that will be applied for all elements at a respective substation. The array follows some order which is indicated bellow. 
#
# `target_configuration = [prod, consp, or_line1, or_line2, ....., ex_line1, ex_line2,..]`

# +
print ('Expected configuration length for SE {} -> {}'.format(sub_id, 
                                                             action_space.get_number_elements_of_substation(sub_id)))

print ('Target configuration array:')
print (target_configuration)

# +
# # ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ 
# Play with this variable to see the change in topology for other nodes

target_configuration[-1] = 1

# # ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ ++ 

new_config = list(target_configuration.astype(int))
print("new_config = ", list(new_config))

# Set new switches to the new state.
action_space.set_substation_switches_in_action(action,
                                               sub_id,
                                               new_config)
print (action)
# -

# ## Visualize the grid
#
# Finally let's display the grid with the action vector.

# +
# Run one step

obs_ary, *_= environment.step(action.as_array())
obs = observation_space.array_to_observation(obs_ary)

# +
'''
The user might specify his/her own layout for every node in the graph and the label position
as well. For a reference layout, please check the visualize_grid.py script.
'''

# Please use the visualize_grid external script to display the 
# eletrical grid in the notebook.
grid_after_action = plot_grid(environment, 
                              obs, 
                              action.as_array())

# Plot the grid
iplot(grid_after_action)

'''
To make a new modification, please do not forget to re-run
the entire notebook again with the desired configuration.
'''
