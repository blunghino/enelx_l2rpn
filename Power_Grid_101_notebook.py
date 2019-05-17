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

# <br><br>
# <h1 style="font-family:'Verdana',sans-serif; color:#1D7874; font-size:30px;">Transmission system operation</h1>

# <br>
# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">Table of Contents</h3>
# <br>
# <ol style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     <li ><a href='#introduction'>Introduction</a></li>
#     <li ><a href='#motivation'>Motivation</a></li>
#     <li><a href='#load_env'>Load an environment with pypownet</a></li>
#     <li><a href='#grid_info'>Grid information and electrical losses</a></li>
#     <li><a href='#grid'>The 4 substation grid</a></li>
#     <li><a href='#line_cut'>A line disconnection and its effect</a></li>
#     <li><a href='#config'>Inside a substation</a></li>
#     <li><a href='#remedial'>A remedial solution</a></li>
#     <li><a href='#conclusion'>Conclusion</a></li>
# </ol>

# <a id='introduction'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">1. Introduction</h3>
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     This notebook is an introduction to power system operations and the Pypownet usage. The operation of electrical power grids have been studied for many years around the world and involves many activities. Power grids, especially transmission grid, aims to transporting electricity over long distances from power plants to consumers. To be efficient and minimize energy losses, power grids often operate under high voltages (above 100kV). One of the main goals for power grid operators is to ensure a reliable and secure continuity of delivery of electricity to consumers under some additional quality criterion that can vary from one country to another.  
# <br><br>
#     For simplicity, a four substations grid is presented here with producers and consumers meshed through a grid. However, transmission grids are more complex and involve many long lines that connect different cities or states. A grid often connects an entire country linking hundreds of power plants and distribution networks but our current example should illustrate clearly enough how congestion could arise on the power grid and which operation could help manage it. Congestion management is actually the most critical and time-consuming task for a power grid operator.
# <br><br> 
#     <img src="https://i.ibb.co/fHm0Rb5/Example-Grid.jpg", width=650, ALIGN="middle", border=20>
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:center; font-size:11px;">
# (courtesy by Marvin Lerousseau)
# </p>
# </p>

# <a id='motivation'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">2. Motivation</h3>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     A power line has some physical capacity, aka thermal limit. Hence there is maximum power flow that is allowed through a line. Beyond it, a power line can break or can induce high risk for the safety of neighboring goods and people. We then say that a power line is congested, as illustrated by the power flow highlighted in yellow above. In that case, operators usually have only a few minutes to reduce the flows in lines. It is very usual for electrical operators to set up the thermal limits above 80% of the line capacity as a warning signal to avoid undesired damages in the equipment.
# <br><br>
#     The first thing operators need to know are the power flows through the lines of the grid to anticipate and monitor congestions. Power flows can actually be computed through physical simulators, under what is called a load flow computation, given productions, consumptions and the topology (aka the graph) of the grid.
# <br><br>
#     As a first intuition, power flows increase when electrical consumption increases. More specifically, the flows in lines are affected by the dynamics of much electrical consumption and the availability of power plants or productions. It often varies along the course of a day, as consumer behaviors and usages change depending on the hour of the day: there are more activities in the middle of the day, while lighting is used mostly overnight and people finally get to sleep, leaving mostly electrical heating systems as well as some few industries in operations.   
# <br><br>  
#     The congestion could be caused by day to day demand, but can also be due to some external factors such as topology grid changes by operator's actions, lines in maintenance or transient faults that can lead to cascading failures or put the system in an instability condition.
# </p>

# <a id='load_env'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">3. Loading the environment</h3>

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     Pypownet is an open-source platform that allows to compute the power flows in a grid and interact as well with a reinforcement learning approach.
# <br><br>
#     Right now, we are going to load all libraries and pypownet environment. The environment contains all essential information about the power grid such as the substations ids where the loads, generators and transmission lines are connected, the power flows and so on.
# </p>

# +
# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from utils.visualize_grid import plot_grid
from matplotlib.font_manager import FontProperties
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Connect Plotly in offline mode. 
init_notebook_mode(connected = True) 
warnings.simplefilter(action='ignore', category=FutureWarning)

# +
import os
import pypownet.environment
import pypownet.runner

# Initialize the env.
environment = pypownet.environment.RunEnv(parameters_folder=os.path.abspath('public_data'),
                                          game_level='4_substations',
                                          chronic_looping_mode='natural', start_id=0,
                                          game_over_mode='hard' )
# -

# <a id='grid_info'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">4. Grid Information and electrical losses</h3>

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# Despide the grid is very small and simple, all information is accessible directly using the pypownet environment. Let's get some information about the most important objects over a power grid: power lines, electrical nodes, loads (aka consumptions) and generators (aka productions). 
# <br><br>
#     The state of those objects are very important in the reinforcement learning challenge as they will be used in observations. It means that given you observe the switch status, power flows, etc the agent should be able to take the correct action to keep flows in normal condition as possible. The following lines of code retrieve the nodes or substations ids and the numbers of elements connected to each busbar. 
# </p>

# +
# Load ids
loads_id = environment.action_space.loads_subs_ids
print ('Num of loads in total: {} -> Loads ids:'.format(len(loads_id), loads_id))
print (loads_id)

# Generators irds
gen_id = environment.action_space.prods_subs_ids
print ('Num of gen in total: {} -> Generators id:'.format(len(gen_id), gen_id))
print (gen_id)

# Line id sender
line_sender_id = environment.action_space.lines_or_subs_id
print ('Total transmission lines: {}'.format(len(line_sender_id)))
print ()
print ('Line sender id:')
print (line_sender_id)

# Line receiver id
line_rcv_id = environment.action_space.lines_ex_subs_id
print ('Line receiver id:')
print (line_rcv_id)
print ()

# Num of elements per SE
print ('Numbers of elements of subestation:')
print ('++  ++  ++  ++  ++  ++  ++  ++  ++')
for i in environment.action_space.substations_ids:
    print ('SE: {:d} - # elements: {}'.format(int(i), 
                                              environment.action_space.get_number_elements_of_substation(i)))

# -

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     Another important factor to control in grid operation is electrical losses. They are present due to some natural electricity physical behavior. They cause an imbalance between productions and consumptions. Losses are more pronounced in long lines because they offer more resistant to the current flows. The necessary condition for a power system to be balanced and hence stable is the following:
# <br><br>
# $$production = consumption + losses$$
# <br>
#     You can get a general idea about losses taking the difference between generators and loads. On Transmission power grid, the amount of losses is often about 2% of the consumption. In Pypownet you can run one step in the environment to get the observation given an action. For now, an action that does not change the status of the switch is passed (action_do_nothing), thus all production and consumptions information are retrieve it.
# <br><br>
# </p>

# +
action_space = environment.action_space
observation_space = environment.observation_space
game = environment.game

# Create do_nothing action.
action_do_nothing = action_space.get_do_nothing_action()

# +
# Run one step in the environment
obs, *_ = environment.step(action_do_nothing)
print (type(obs))

# for key, value in obs.items():
#      print(key, value)
# -

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# As you see the observation is a numpy array. Sometimes is better to transform it as observation calling the following method with pypownet.
# </p>

obs = observation_space.array_to_observation(obs)
vars(obs)

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# The observation contains all essential information about the grid. There are two ways you could compute the electrical losses. Tne first mehotd is indicated below. The losses are no more than the difference between what gets out from origin nodes and what gets into the extremity ones.

# Losses in transmission lines.
losses_in_lines = np.abs(obs.active_flows_origin) - np.abs(obs.active_flows_extremity)
print ('Losses in transmission lines {} MW'.format(losses_in_lines))
print ('Total sum losses in lines {:.4} MW'.format(sum(losses_in_lines)))

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# The same value could be computed taking into account the formula described above.
# </p>

print ('Total production {:.4} MW'.format(sum(obs.active_productions)))
print ('Total consumption {:.4} MW'.format(sum(obs.active_loads)))
print ('Losses in grid (prod - consump) {:.4} MW'.format(sum(obs.active_productions) - sum(obs.active_loads)))

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# Once reviewed how to obtain the grid information, It is time to execute actions. An action vector should be initialized beforehand. The vector will be modified it later with the desire action applied to a particular switch status in substations or lines.
# </p>

# Initialize applied action
applied_action = action_space.get_do_nothing_action(as_class_Action=True)
print (type(applied_action))

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# The next function allows us to perform a few iterations using the step method in the environment given an action as an input and returns the observation variables as the power flow results, switches states, etc.
# </p>

# +
import copy

def sim(action, 
        t_action=0):
    
    # Restart all the game from the scratch.
    env = []
    env = pypownet.environment.RunEnv(parameters_folder=os.path.abspath('public_data'),
                                          game_level='4_substations',
                                          chronic_looping_mode='natural', start_id=0,
                                          game_over_mode='hard') 

    observation_space=env.observation_space
    
    # Iterating process..
    for i in range(1):
    
        # Execute action at step 0.
        if i == t_action: 
            obs_arry, *_ = env.step(action)
            obs = observation_space.array_to_observation(obs_arry)
            obs_action = copy.deepcopy(obs)
    
    return env, obs_action


# -

# <a id='grid'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">5. A simple grid</h3>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# The four substation grid is presented as follows. All power plants, consumptions and lines are on service and the load flows are indicating in each substation.
# </p>

# <div class="alert alert-warning">
#     Note: How to interprate the flows?... for transmission lines (or) stands for origin and the convention is to take the flow as positive while (ex) means extremity and should have taken as negative. Thus one might guess the power flow direction.
# </div>

layout = {'1':(-10, 10), '2':(10, 10), '3':(-10, -10), '4':(10, -10)}
label_pos = {'1':'top left', '2':'top right', '3':'bottom left', '4':'bottom right'}

# +
# Run the game.
n_iter = 1

# Run the environment.
env, obs = sim(action_do_nothing)

# Get the grid.
grid_do_nothing = plot_grid(env, obs, action_do_nothing, 
                           layout, label_pos, size=(45,45))
iplot(grid_do_nothing)
# -

# <h4 style="font-family:'Verdana',sans-serif; color:#1D7874;">How to interpret the results?</h4>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# The values of the power flows are attached to each node. As we mentioned earlier, the direction has to be taken positive if it leaves the substations and negative otherwise. For instance, the direction of the flow in the line 1-4 gets out the substation 1 and gets in substation 4. 
# </p>

# <a id='line_cut'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">6. Line disconnection</h3>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     Transmission lines link different states or regions between productions and consumptions. They have long distance and are subjected to suddenly disconnections due to thunderstorms. It is natural to have one or more lines out-of-service along the day because of thunder hits a line. This number tends to increase during winter. 
# <br><br>
#     A disconnected line changes the configuration on the grid and causes a power flow's redistribution in the grid. Sometimes these undesired events lead to some overloads in other lines. You have not guess the reason why yet? The answer is very intuitive. When a line get disconnected, the same amount of energy is still demanded by customers, but now you have one path less in the grid to transport the energy, and the others will hence be loaded with more power.
# <br><br>
#     Substations are equipped with switches to disconnect a line when a fault or congestion is detected. Normally the criteria to disconnect a line under a congestion is to measure the flow and the time a line remains congested above a given threshold the switch automatically open a line.
# <br><br>
#     This behavior occurs many times in electrical systems and the expertise and vast knowledge of operators help to alleviate the stress in order to guarantee energy to customers. A similar action could be replicated using pypownet. We simulate a disconnection in our small grid to see the resultant state of the grid.
# </p>

# <div class="alert alert-warning">
#     Note: The user can modify and play with other lines.
# </div>

# Specify the line to be disconnected.
id_line = 0

# +
# Initialize action class
applied_action = action_space.get_do_nothing_action(as_class_Action=True)

# Apply the action (change switch status of a line)
action_space.set_lines_status_switch_from_id(action=applied_action,
                                             line_id=id_line,
                                             new_switch_value=1)

print ('New action vector')
print (applied_action.as_array())
# -

# <h4 style="font-family:'Verdana',sans-serif; color:#1D7874;">How the action vector is compossed?</h4>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     The action vector is no more than a concatenation over all arrays allowed to change the status to each element (on/off) in the grid. By default all elements are online before running the game. In the following figure you have a clear representation of the action vector for this particular case.
# <br><br>
#     <img src="http://i66.tinypic.com/2vmj682.jpg", width=850, ALIGN="middle">
# <br><br>
#     But what really is an action ? An action in pypownet emulates what electrical operators do in control rooms. They monitor the grid in real time and then execute commands in a HMI (Human-Machine Interface) that are sent them to substations to open or close switches. The following figure try to illustrate this principle. 
# <br><br>
#     <img src="http://i67.tinypic.com/3008ar7.png", width=700, ALIGN="middle">
# </p>

# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# Once the action vector has been created, let's run the game to see how the grid is affected.
# </p>

# +
# Convert action vector Class into array.
applied_action = applied_action
# Run simulation
env, obs = sim(applied_action)

# Plot the grid.
grid_do_nothing = plot_grid(env, obs, applied_action, 
                            layout, label_pos, size=(45,45))
iplot(grid_do_nothing)
# -

# <h4 style="font-family:'Verdana',sans-serif; color:#1D7874;">Analyzing results...</h4>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
# In the previous results, we simulated a line disconnection caused by a lightning in the path 1-2. You immediately may notice some lines have changed the color. Overload events have been detected over the lines 1-4 and 1-3. The resulting power flow, after the redistribution, causes a  flow increments over their thermal limits. In such condition, electrical operators have a few minutes to execute remedial actions to alleviate the system.
# </p>

# <a id='config'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">7. Inside a Substation</h3>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     A power grid is composed of power lines connected through substations. Within a substation, there can be several arrangements between branches (lines, generators and loads) that allow them to meet. There exist many predefined configurations and each of them are better than others in terms of reliability.
# <br><br>
#     In pypownet, substations have "double busbar layout", which means that you can make no more than two electrical nodes per substations. The double busbar configuration consists of two bars on which you can connect together generators, loads and lines. For each of those elements, you have to choose the busbar it is connected to at a given time. The following figure is a representation of such configuration.
# <br><br>
#     <img src="http://i65.tinypic.com/9s6y2s.png", width=800, ALIGN="middle">
# <br><br>
#     You can see in the first diagram is what is inside in a substation with a double busbar configuration. Regularly, all busbars are connected to each other. An action could split the busbars and pass some elements to the desired bar, thus a flow reconfiguration is possible.
# </p>

# <a id='remedial'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">8. A Remedial solution</h3>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     When an overload is detected, operators have to execute a remedial action to alleviate the stress in the system. Normally the experience gained by years by people who work in electrical control rooms is a crucial factor to determine the best one. 
# <br><br>
#     Transmission networks are a bit complex with many paths where the power can flow (mesh grids). This also gives many alternatives to solve the congestions. In many countries changing the production is a valid solution. For instance, you can decrease the flow in the generators which are connected to the overloaded line and increase others that are near the consumptions.
# <br><br>
#     However, the previous one is costly, especially in countries where producers are market actors. An always cheap one is to play with the main grid configuration, thus one does not have to change the productions. These are the kind of actions Rte wants to use to build a new smart controllers for the power grid through the challenge. 
# <br><br>
#     To illustrate the issue, let's imagine the flow in the line 1-4 and 1-3 have increased by some external factor (such a lightning in the line 1-2). How one would solve the congestion with topological reconfiguration? Actually, one solution is to split the substation 4 and connect the line a selective path 1-2-4. Thus, a fictitious double line is created to supply the energy for load in substation 4. This kind of game is the one that the agent should be able to learn after analyzing different scenarios.
# <br><br>
#     This action can be modeled with pypownet as well. In the next lines of code, you can see all the steps needed to perform the desired action and to get the grid with the power flow results. A new configuration layout should be provided by the user. For example, in the example described below, we execute an action to split the node 4 and connected the extremity with the line that comes from the node 1, line 2 and the respective load.
# </p>

# +
# SE id we want to modify.
sub_id = 4

# Change internal configuration of SE.
new_switch_config = [1, 1, 1, 0]
# -

# Initialize action class
applied_action = action_space.get_do_nothing_action(as_class_Action=True)

# +
#  Set new switches to the new state.
applied_action.set_substation_switches(sub_id, new_switch_config)

# See changes.
sub_i_new_config, sub_i_elm_type = applied_action.get_substation_switches(sub_id)
print ('New configuration of SE: {}'.format(sub_id))
print ('++   ++   ++   ++   ++   +')
for switch_status, elm_sub_type in zip(sub_i_new_config, sub_i_elm_type):
        print ('({}, {})'.format(switch_status, elm_sub_type.value))
# -

# <div class="alert alert-block alert-danger">
# <b>Hint</b> Please be aware the method set_substation_switches set the switches in the grid with the desire configuration. If one executes a new step in the environment, the grid will remain in the same configuration as specified in the new_switch_config. For the RL game is recommended to use the method action_space.set_substation_switches_in_action which emulates the on/off action.
# </div>

# +
# Run the environment.
env_after_action, obs_after_action = sim(applied_action.as_array())


# Plot the grid.
layout = {'1':(-10, 10), '2':(10, 10), '3':(-10, -10), '4':(5, -10),'6664':(10, -10)}
label_pos = {'1':'top left', '2':'top right', '3':'bottom left', '4':'bottom center','6664':'bottom right'}

grid_after_action = plot_grid(env_after_action, 
                              obs_after_action, 
                              applied_action.as_array(), 
                              layout, label_pos, size=(45,45))
iplot(grid_after_action)
# -

# <a id='conclusion'></a>

# <h3 style="font-family:'Verdana',sans-serif; color:#1D7874;">9. Conclusion</h3>
#
# <p style="font-family:'Verdana','sans-serif'; color:#393D3F; text-align:justify; font-size:14px;">
#     In this notebook, we reviewed very quickly the main operation principles in transmission networks with a small grid. The continuity of the electric power service is crucial for nowadays economy and should be robust enough to any hazards or atypical sistuations in its operations. A undesired line disconnection without an effective action that could solve the problem might bring blackouts and economic losses. 
# <br><br>
#     As we mentioned earlier, transmission operation involves many tasks but all of them are met to have one single objective: supply reliable and secure electricity. Operators work every day to minimize the impact of external factors that put the electrical system at risk but eventually when the grid is large and meshed, the daily operation is more complicated.
# <br><br>
#     The Pypownet platform was designed to interact together with the reinforcement learning approach. It allows executing actions in the transmission grid to manage lines with congestion without any cost involved. We covered all actions such as line disconnection and node splitting in the respective section. A user could play and execute more complicated ones through combinations.  
# <br><br>
#     The end state during the challenge is to train an agent able to learn a policy that could overcome obstacles, aka congestions, while optimizing the line usage rates to reflect a real operation in transmission grids.
# </p>


