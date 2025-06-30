This is Bike Map, also known as Mafia Map.

The main goal of this software is to make it safer to ride a bicycle
speed up your bus, make it safer to be a pedestrian and create a nonpartisan
method for analyzing the safety and comfort of streets.

This program makes extreme usage of GPUs and was designed specifically for a 
computer with the following qualities: 2 Nvidia 5060 Ti GeForce RTX GPU 
with 16 GB of VRAM each, 128 GB of RAM, 100+ TB of hard drive space,
16 core AMD Ryzen core. This software is designed for a production setting 
and features several additional features such as a powersafe programming model
that saves processing and continues where it last left off.

This program can be used on any computer but this program makes extreme usage
of video decoders in the GPUs and tensor threads in the GPUs to accelerate the
program by more than 2700x (or more) faster. Meaning on my system I am able to process 
about 40-50 frames per second, but on a standard computer like a standard desktop
it is likely able to process about .01-.04 frames per second or less.
But whatever your system is, it can likely rin this program.

This software constitutes basically 2 programs: Matcher and Visualizer.

The most current stable version of the Matcher program is Matcher51.py
this program has a specific job, collect GPX and MP4 files from a single directory
and then watch the video and look at the GPX files and determine how likely the video 
and gpx files match together to get the correct video timestamps.
In Matcher50 there is only a comparison between speed and acceleration between video and gpx.
In Matcher51 there have been the addition of several new dimensions for the tensor comparison.
The new dimensions in the Matcher51 include, sunlight tracking, sun positioning, greenery, 
number of buildings and urbanism, pavement, elevation changes, shakiness of camera movement
indicating dirt of off roading.

Using only the acceleration and speed of the camera, Matcher50 is about 76% accurate.

Matcher50 and beyond specifically are designed to be able to process both FLAT and 360 panoramic
videos.

After running Matcher, there is an output file that is very large file ending in ramcache.json.
This is the main output file, which then gets sent to Visualizer.

Visualizer makes heavy usage of Ai and literally watches the videos.
Visualizer makes very large use of YOLO and Computer Vision to detect and organize what is happening in the videos,
it also is able to listen to the audio of the videos and is able to make a sound heat map.
Visualizer so far is designed to do the following:

Count humans (such as pedestrians or people sitting)
Count Cars
Count Bicycles (separately from humans)
Count Buses
Detect Stoplights
Record the phase of the stoplight (such as red/yellow/green)
Record the direction and speed of humans, bicycles, vehicles

These metrics are collected as the City of Minneapolis failed to respond to FOIA requests,
and then proceeded to stall records requests in Court to allow for the natural 
deletion of camera footage. So it became necessary to design a system that could be used by anyone
and by all. 

The next step of this project is add the following features that begin to make this a little creepy,
and thus the new name of "Mafia Map". Many in the bicycle community want stronger 
reprocussions against unsafe drivers and illegal driving. The next item to be added to this system is:
the ability to see inside of a vehicle (if the video resolution is high enough) by using Ai facial recognition
and changes to image contrast and referencing social media, if the license plate is read aloud by the user.
Also the ability to record license plates to detect outdated tabs.
Also the ability to recognize bike lanes,
ability to detect bus stops and bus lanes,
detect illegal parking in bus stops, bus lanes, bike lanes, etc.

Further additions that are possible include the ability to store faces and detect them in real time in real life,
also the ability to detect problem individuals such as cars known for trying to run people over 
and issue a real time alert to a user of the vehicle's proximity, among many other features.

This project is completely open source and can be used in any manner that someone wants for any purpose.
