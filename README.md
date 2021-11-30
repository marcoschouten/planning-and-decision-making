# Planning_Project
RO47005 Planning &amp; Decision Making


* https://www.overleaf.com/project/6196af29a03b1320edb061ef
* https://www.overleaf.com/project/6196adfebe6f0071ad8f0136



Task Division

#### Task1: World Building, Simulation, Map 
Chenghao
 <ul>
  <li>Input: complexity (size and number of obstacles)</li>
  <li>Output: Map</li>
</ul>
Workflow steps:
<ol>
  <li>generate a hardcoded static environment (so that other team mates can test their algorithms)</li>
  <li>generate a randomizer i.e.  given some input parameters (complexity of map 0,1 value), generate maps that have reachable paths from start to goal.</li>
  <li>transfer this code to ROS</li>
</ol>


 #### Task2: Generate High level path 
 Marco
<ul>
  <li>Input: map</li>
  <li>Output: path, List (x,y,z)</li>
</ul>
RRT* and variants, with steering function




 #### Task3 Generate Trajectory
 Stan
<ul>
  <li>Input: path List (x,y,z)</li>
  <li>Output: trajectory, List (t,x,y,z)</li>
</ul>


 #### Task4: Controller to move the robot
 Brandon
<ul>
  <li>Input: Trajectory, List (t,x,y,z)</li>
  <li>Output: Robot's controls (u) and movment in the simulation </li>
</ul>
