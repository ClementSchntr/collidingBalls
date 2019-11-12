# CollidingBalls
This program displays moving balls that will collide with each other in a new window.

## Prerequisites
Running this programm requires Python 3 as well as the modules tkinter, time, math, random and numpy.

## How to run
To see this execute the play.py file. Doing so requires the tkinter module for Python 3.

When executing play.py the result is not being computed in real time. It is playing back the recording contained in the file recBall1.txt. Computing the moves in advance makes it possible to perform much more involved computation, while the movements remain smooth and free from stutter.

To create your own recording file execute collidingBalls_precomp.py. It will then overwrite recBall1.txt.
