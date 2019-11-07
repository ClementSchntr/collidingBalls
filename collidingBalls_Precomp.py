# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:17:53 2018

@author: Clément
"""
#import tkinter
#from tkinter import Tk, Canvas
import random, time, numpy as np, math
#from collisionSim_Playback import play
from scipy.optimize import fsolve

"""
To Do:
    -frametime plot
    -fragment polygons into smaller polygons
"""

#canvas size
WIDTH = 1000
HEIGHT = 700

fileName = "recBall1.txt" # Name of the record-file that will be created

targetFPS = 61
targetFrametime = 1 / targetFPS

frame_upperBound = 1 * 60 * 60 # number of frames to be computed

objectsizeMax = 80
objectsizeMin = 30
columns = math.floor(WIDTH/objectsizeMax)
rows = math.floor(HEIGHT/objectsizeMax)
maxBalls = rows * columns
print("maxBalls: " + str(maxBalls))

noObjects = 50 # maxBalls # Number of balls on the canvas
print("noObjects: " + str(noObjects))
initialState = 0 # 0 = balls have random startpos and random velocity vectors


collision = True #set to true to enable collision between balls

#gravity = False #set to true to enable gravity
mass = False # set to True for mass to be taken into account in collisions

#gravityStrength = 1 #unit of acceleration: pixel/frame^2
velocityScale = 3 # scaling factor for the maximum start velocity of each ball

Inf = float("inf") # positive infinity

#tk = Tk()
#fenster = Canvas(tk, width=WIDTH, height=HEIGHT, bg="white")
#tk.title("The night is dark and full of terrors")
#fenster.pack()

rec = open(fileName,'w') 


def randWithoutZero(min, max): # returns a uniformly distributed integer i!=0 between min and max
    if min==max:
        return min        
    if max < min:
        return randWithoutZero(max, min)
    l = list(range(min, max + 1))
    if 0 in l:
        l.remove(0)
    return random.choice(l)
        
def normalize(v): # length of v is changed to 1
    return v / np.linalg.norm(v)

def angle(v1, v2):
    arccosInput = np.dot(v1, v2)/ np.linalg.norm(v2)/ np.linalg.norm(v1)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    theta = np.arccos(arccosInput)/math.pi
    R = np.array([[0,-1],[1,0]]) # rotation by pi/2
    perpV1 = np.dot(R,v1)
    if np.dot(v2, perpV1) < 0:
        theta *= -1
    return theta


class Ball:
    def __init__(self, color, startPos, speed, size):
        self.color = color
        self.size = size # diametre in pixels
        self.coords = np.array([startPos[0] - self.size /2, startPos[1] - self.size /2, 
                                startPos[0] + self.size /2, startPos[1] + self.size /2])
        if mass:
            self.mass = math.pi * (self.size / 2)**2 # the mass of the ball uniformly distributed, 
            #therefore mass is proportional to area. Mass will be measured in pixel^2
        else: self.mass = 1
        
#        self.shape = fenster.create_oval(startPos[0]- self.size /2 , startPos[1] - self.size /2, 
#                                         startPos[0] + 0.5 * self.size, startPos[1] + self.size / 2, fill=self.color)
        self.speed = speed 
        toWrite = "cb " + color + " "
        for c in self.coords:
            toWrite += str(c) + " "
        rec.write(toWrite + "\n")
        
    def move(self, t): #moves the ball by t * speed and adjusts the velocity vector if the ball hits the frame
        
        pos = self.coords
        s = self.speed
        v = np.array([s[0],s[1],s[0],s[1]])
#        print(v)
        pos = pos + v
        self.coords = pos
        rec.write("mb " + str(balls.index(self)) + " " + toString(pos) + "\n")
        
        if (pos[3] >= HEIGHT and self.speed[1] > 0) or (pos[1] <= 0 and self.speed[1] < 0):
            self.speed[1] = -self.speed[1]

        if (pos[2] >= WIDTH and self.speed[0] > 0) or (pos[0] <= 0 and self.speed[0] < 0):
            self.speed[0] = -self.speed[0]

            
    def centre(self): # returns the coordinates of centre of the circle
        pos = self.coords
        return np.array([pos[0] + 0.5 * self.size, pos[1] + 0.5 * self.size])
    
    


        

def toString(l): # [1,2,3] --> "1 2 3 "
    s = ""
    for n in l:
        if n == 0:
            length = 1
        else:
            length = np.log10(np.abs(n)) #number of digits before the decimal dot
            length = int(np.floor(length))
        if length < 0:
            length = 0
        minusSign = 0
        if n < 0:
            minusSign = 1
        s += str(n)[0:length + 3 + minusSign] + " "
    return s

def degToRad(d): # convert degrees into radians
    return d * math.pi / 180
    
def toVector(pos): #take list of numbers of length 2n and returns a list of vectors of length n
    lov =[]
    for i in range(math.floor(len(pos)/2)):
        lov.append(np.array([pos[2*i],pos[2*i+1]]))
    return lov
    
def rotMat(theta): # 2x2 rotation matrix of angle theta
    return np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    
def ballStartPos(i):
    row = math.floor(i/columns)
    column = i % columns
    return np.array([(column + 0.5) * objectsizeMax, (row + 0.5) * objectsizeMax])

def createBall():
    color = random.choice(colorsCopy)
    colorsCopy.remove(color)
    i = len(balls)
    size = random.randrange(objectsizeMin, objectsizeMax)
    startPos = ballStartPos(i)
    speed = np.array([randWithoutZero(-50, 50)/100 * velocityScale, randWithoutZero(-50, 50)/100 * velocityScale])
    return Ball(color, startPos, speed, size)
    
def col(i, j): # process collision between ith and jth ball
    if j < 0 :
        pol = balls[i]
        
#        pol.speed *= -1
        vert = pol.vertices()
        k = colCorners[int(i),int(j+4)]
        colDir = normalize(vert[int(k)])
#    colDir = normalize((balls[j].centre() - balls[i].centre())) # normed direction vector from i to j
        perpDir = np.dot(np.array([[0,-1],[1,0]]) ,colDir) # normed vector perpendicular to colDir. Optained by pi/2 rotation

        colVelocity = np.dot(colDir, pol.speed) 
        perpVelocity = np.dot(perpDir, pol.speed)
        if colVelocity > 0 :
            pol.av = - pol.av
            pol.R = rotMat(pol.av)
            pol.speed = (perpVelocity * perpDir - colVelocity * colDir)
#            balls[j].speed = (perpVelocity[1] * balls[j].mass * perpDir + colVelocity[0] * balls[i].mass * colDir) / balls[j].mass
    else:
        colDir = normalize((balls[j].centre() - balls[i].centre())) # normed direction vector from i to j
        perpDir = np.dot(np.array([[0,-1],[1,0]]) ,colDir) # normed vector perpendicular to colDir. Optained by pi/2 rotation
        colVelocity = [] 
        perpVelocity = [] 
        for b in [balls[i], balls[j]]: # decomposition of velocity vectors into 'collision' and 'perpendicular' components
            colVelocity.append(np.dot(colDir, b.speed))
            perpVelocity.append(np.dot(perpDir, b.speed))
        if colVelocity[0] > colVelocity[1] :
            balls[i].speed = (perpVelocity[0] * balls[i].mass * perpDir + colVelocity[1] * balls[j].mass * colDir) / balls[i].mass
            balls[j].speed = (perpVelocity[1] * balls[j].mass * perpDir + colVelocity[0] * balls[i].mass * colDir) / balls[j].mass

    
    



            



colors = ['red', 'green', 'blue','midnightblue','black', 'white', 'firebrick', 'navy', 'orangered', 
          'orange', 'yellow', 'magenta', 'dodgerblue', 'darkviolet',  'darkorange', 'forestgreen', 
          'grey', 'pink', 'magenta', 'brown']

colorsCopy = colors.copy()
          
balls = [] # stores all the balls on the canvas

if initialState == 0:
    for i in range(noObjects):
        if len(colorsCopy) == 0:
            colorsCopy = colors.copy()
        balls.append(createBall())
elif initialState == 1:
    mass = True
    balls.append(Ball('red', [100,200], np.array([-2,0]), 100))
    balls.append(Ball('blue', [500,200], np.array([-2,0]), 50))
    balls.append(Ball('green', [600,500], np.array([0,0]), 100))
    balls.append(Ball('violet', [150,500], np.array([4,0]), 100))
    balls.append(Ball('yellow', [700,500], np.array([0,0]), 100))
    balls.append(Ball('firebrick', [800,500], np.array([0,0]), 100))
    balls.append(Ball('lightblue', [400,500], np.array([0,0]), 100))
elif initialState == 2:
    mass = True
    balls.append(Ball('red', [300,350], np.array([4,0]), 100))
    balls.append(Ball('green', [700,350], np.array([-4,0]), 100))
    balls.append(Ball('blue', [500,350], np.array([0,0]), 100))
    balls.append(Ball('violet', [500,550], np.array([0,-4]), 100))
    balls.append(Ball('yellow', [500,150], np.array([0,4]), 100))
#    balls.append(Ball('firebrick', [800,200], np.array([-2,0]), 100))
    
    
noObjects = len(balls)



frameCounter = 0
percentCounter = 0 # helps calculating the percentage frames that have already been computed
t0 = time.time()
t1 = 0 # timestamp of the last frame
t2 = 0 # time passed since t0
t3 = 0
colCounter = 0
computeSum = 0
maxComputeTime = 0
minFPSuncapped = Inf #100000

#additionToColList = 0
colList = [] # list of tuples (i,j). Stores the object pairs(indices) that will collide before the next frame
toMove = [1] * noObjects # stores how much the balls have to be moved after collision processing. Unit: frame (time)

distMat = np.zeros((noObjects,noObjects)) # distance matrix: store distances between ball centres
# Only populated in the lower triangle
# The index to ball association is the same as in the list 'balls'

            
colPro = 0 #collision processing counter
# preprocessing



while frameCounter <= frame_upperBound:
    if collision : # collision is a boolean
        for i in range(len(balls)):
            for j in range(i):
                distMat[i,j] = np.linalg.norm(balls[i].centre() - balls[j].centre())
                if distMat[i,j] <= 0.5 * (balls[i].size + balls[j].size) :
                    col(i, j)
    for ball in balls:
        ball.move(1) 

    rec.write("ef\n")

    toMove = [1] * noObjects


    frameCounter += 1
    if frameCounter / frame_upperBound * 100 >= percentCounter:
        print(str(percentCounter) + "%")
        percentCounter += 1
    #t1 = t2
    t2 = time.time() - t0 # time passed since t0
    computeTime = t2 - t3
    
    if computeTime > maxComputeTime:
        maxComputeTime = computeTime
        minFPSuncapped = 1 / maxComputeTime
    
    computeSum = computeSum + computeTime
    avgComputeTime = computeSum / frameCounter
    avgFPSuncapped = 1 / avgComputeTime
    avgColperFrame = colCounter / frameCounter 
    
    
    t3 = time.time() - t0
    avgFPS = frameCounter / t3 # running average
    
rec.close()
print("completed\n"+str(frameCounter)+ " frames were computed in " + str(t3)[:math.floor(abs(math.log10(t3)))+2] + " seconds")

#play()
