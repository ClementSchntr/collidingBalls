# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:17:53 2018

@author: Clément
"""
#import tkinter
#from tkinter import Tk, Canvas
import random, time, numpy as np, math
#from collisionSim_Playback import play
#from scipy.optimize import fsolve

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

noObjects = 60 # maxBalls # Number of balls on the canvas
print("noObjects: " + str(noObjects))
initialState = 0 # 0 = balls have random startpos and random velocity vectors


collision = True #set to true to enable collision between balls
displayColtime = True

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
        
    def move(self, i, t): #moves the ball by t * speed and adjusts the velocity vector if the ball hits the frame
#        fenster.move(self.shape, t * self.speed[0], t * self.speed[1])
        
        pos = self.coords
        s = self.speed
        v = np.array([s[0],s[1],s[0],s[1]])
#        print(v)
        pos = pos + v
        self.coords = pos
        rec.write("mb " + str(balls.index(self)) + " " + toString(pos) + "\n")
        if (pos[3] >= HEIGHT and self.speed[1] > 0) or (pos[1] <= 0 and self.speed[1] < 0):
            self.speed[1] = -self.speed[1]
            recomputeColTime(i,t)
        if (pos[2] >= WIDTH and self.speed[0] > 0) or (pos[0] <= 0 and self.speed[0] < 0):
            self.speed[0] = -self.speed[0]
            recomputeColTime(i,t)
            
    def centre(self): # returns the coordinates of centre of the circle
        pos = self.coords
        return np.array([pos[0] + 0.5 * self.size, pos[1] + 0.5 * self.size])
    
    

class Polygon:
    def __init__(self, v, av, color, *p):
        self.color = color
#        self.shape = fenster.create_polygon(p, fill=color)
        self.coords = p # list of corners
        self.speed = v # velocity in pixels per frame
        self.av = av # angular velocity in radians per frame
        self.R = rotMat(av)
        self.smallRadius = -1 #distance from G to the closest corner
        self.outerRadius = -1 #distance from G to the corner furthest away
        self.mass = 1
#        n = len(p)
#        if n < 10:
#            n = "0" + str(n) + " "
        toWrite = "cp " + color + " "
        for c in p:
            toWrite += str(c) + " "
            
#        toWrite += " c" + color
        rec.write(toWrite + "\n")
    
    def vertices(self): # returns list of G-to-vertex vectors sorted clockwise by angle starting at np.array([-1,0])
        pos = self.coords # this is a list
        vert =[]
        G = self.centre()
        for i in range(math.floor(len(pos)/2)):
            vert.append(np.array([pos[2*i],pos[2*i+1]]) - G )
        if self.outerRadius == -1:
            self.outerRadius = np.linalg.norm(sorted(vert, key = np.linalg.norm)[len(vert)-1])
#            print(vert)
#            print(self.outerRadius)
        e1 = np.array([1,0])
        f = lambda v : angle(e1,v)
        vert = sorted(vert, key = f)
        return vert
    
    def centre(self): # returns isobarycentre of the vertices
        pos = self.coords
        vert = toVector(pos)
        G = np.array([0.0,0.0])
        for v in vert:
            v = v #* 1/len(vert)
            G += v
        G = G/len(vert)
        return G
#        return np.array([0,0])

    def move(self, i, t):
#        if self.av == 0:
##            fenster.move(self.shape, t * self.speed[0], t * self.speed[1])
#            p = self.coords
#            for i in range(len(p)):
#                p[i]= p[i] + t * self.speed
#            self.coords = p
#            rec.write("mr " + str(t * self.speed) + " 0\n") # mr = move rotate
#        else:
#            pos = fenster.coords(self.shape)
        G = self.centre()
        vt = t * self.speed
        G = G + vt
        vert = self.vertices()
        p = []
        r = self.R
        for v in vert:
            w = np.dot(r,v) + G
            p.append(w[0])
            p.append(w[1])
#            fenster.delete(self.shape)
#            self.color = random.choice(colors)
#            self.shape = fenster.create_polygon(p, fill= self.color)
        self.coords = p
#        for i in range(len(p)):
#            if i%2 == 0:
#                if p[i] >= HEIGHT:
#                print(frameCounter)
        [xmin, v_xmin, ymin, v_ymin, xmax, v_xmax, ymax, v_ymax] = boundaries(p)
        if xmin <= 0:
            colCorners[i,3] = v_xmin
            col(i,-1)
        if ymin <= 0:
            colCorners[i,0] = v_ymin
            col(i,-4)
        if xmax >= WIDTH:
            colCorners[i,1] = v_xmax
            col(i,-3)
        if ymax >= HEIGHT:
            colCorners[i,2] = v_ymax
            col(i,-2)
        rec.write("mp " + str(balls.index(self)) + " " + toString(p) + "\n")

def boundaries(p): # computes the two smallest intervals within which the coordinates of p lie
    xmin = Inf
    v_xmin = -1
    xmax = -Inf
    v_xmax = -1
    ymin = Inf
    v_ymin = -1
    ymax = -Inf
    v_ymax = -1
    for i in range(len(p)) :
        if i%2 == 0 :
            if p[i] < xmin:
                xmin = p[i]
                v_xmin = math.floor(i/2)
            if p[i] > xmax:
                xmax = p[i]
                v_xmax = math.floor(i/2)
        elif i%2 == 1:
            if p[i] < ymin:
                ymin = p[i]
                v_ymin = math.floor(i/2)
            if p[i] > ymax:
                ymax = p[i]
                v_ymax = math.floor(i/2)
    return [xmin, v_xmin, ymin, v_ymin, xmax, v_xmax, ymax, v_ymax]
        

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
#    startPos = [random.randrange(0 + math.floor(size/2)+1, WIDTH - math.floor(size/2)), 
#                random.randrange(0 + math.floor(size/2)+1, HEIGHT - math.floor(size/2))]
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

def colTime(i, j): # returns the number of frames until the i-th and j-th object collide. 
    # returns infinity if there is none 
    if type(balls[i]) == Ball and type(balls[j]) == Ball:
        v = balls[i].speed - balls[j].speed
        pos = balls[i].centre() - balls[j].centre()
        
        a = np.linalg.norm(v) ** 2
        b = 2 * np.dot(pos, v)
        c = np.linalg.norm(pos) ** 2 - ((balls[i].size + balls[j].size)/2) ** 2
        delta = b ** 2 - 4 * a * c
        
        if delta > 0 :
            result = (-b - math.sqrt(delta))/(2*a)
            if result >= 0 :
                return result
            else :
                return Inf
        else:
            return Inf
    elif type(balls[i]) == Polygon and type(balls[j]) == Polygon:
        """ """

def borderColTime(i):
    pol = balls[i]
    e1 = np.array([1,0])
    times = np.array([Inf,Inf,Inf,Inf])
    corners = np.array([-1,-1,-1,-1])
    vert = pol.vertices()
    G = pol.centre()
    outRad = pol.outerRadius
    av = pol.av
    (vx,vy) = pol.speed
    
    for i in range(len(vert)):
        v = vert[i]
        r = np.linalg.norm(v)
        theta =angle(e1,v)
        t = 4 * [0]
        t[0] = verticalColTime(0,G[1],vy,av,theta,r,outRad)
        t[1] = horizontalColTime(WIDTH,G[0],vx,av,theta,r,outRad)
        t[2] = verticalColTime(HEIGHT,G[1],vy,av,theta,r,outRad)
        t[3] = horizontalColTime(0,G[0],vx,av,theta,r,outRad)
        for j in range(4):
            if t[j] < times[j]:
                times[j] = t[j]
                corners[j] = i
    return (times,corners)
    
def horizontalColTime(W,Gx,vx,av,theta,r,outRad):
    if vx == 0:
        return Inf
    f = lambda t : math.cos(theta + t * av)*r + Gx + t * vx -W
    t0 = (W - outRad - Gx)/ vx
    return fsolve(f,t0)

def verticalColTime(H,Gy,vy,av,theta,r,outRad):
    if vy == 0:
        return Inf
    f = lambda t : math.sin(theta + t * av)*r + Gy + t * vy -H
    t0 = (H - outRad - Gy)/ vy
    return fsolve(f,t0)

#    return horizontalColTime(H,Gy,vy,av,theta - math.pi/2,r,outRad)

def val(tup):
    (i,j) = (tup[0], tup[1])
    if j < 0:
        return borderColTimeMat[i,j+4]
    else:
        return colTimeMat[i,j]

def sortByColTime(l):
    l = sorted(l, key = val)
    #return l

def recomputeColTime(i, t): # computes the collision times of i with all others and updates colList if needed
#    (borderColTimeMat[i],colCorners[i]) = borderColTime(i)
#    borderColTimeMat[i] += np.array([t,t,t,t])
    for k in range(noObjects):
        c = 2
        if k > i:
            c = colTime(k,i) + t
            colTimeMat[k,i] = c
#            if c < 1:
#                colList.append((k,i))
        if k < i:
            c = colTime(i,k) + t
            colTimeMat[i,k] = c
#            if c < 1:
#                colList.append((i,k))
#        sortByColTime(colList)
            


def updateColList(): # duplicate-free list of collisions to happen before the next frame
    colList = []
    for i in range(len(balls)):
#        for j in range(4):
#            if 0 <= borderColTimeMat[i,j] < 1:
#                colList.append((i,j-4))
        for j in range(i):
            if 0 <= colTimeMat[i,j] < 1 :
                colList.append((i, j))
    sortByColTime(colList)
    return colList


colors = ['red', 'green', 'blue','midnightblue','black', 'white', 'firebrick', 'navy', 'orangered', 
          'orange', 'yellow', 'magenta', 'dodgerblue', 'darkviolet',  'darkorange', 'forestgreen', 
          'grey', 'pink', 'magenta', 'brown']

colorsCopy = colors.copy()
          
balls = [] # stores all the balls on the canvas

if initialState == 0:
    displayColtime = False
    for i in range(noObjects):
        if len(colorsCopy) == 0:
            colorsCopy = colors.copy()
        balls.append(createBall())
elif initialState == 1:
    displayColtime = True
    mass = True
    balls.append(Ball('red', [100,200], np.array([-2,0]), 100))
    balls.append(Ball('blue', [500,200], np.array([-2,0]), 50))
    balls.append(Ball('green', [600,500], np.array([0,0]), 100))
    balls.append(Ball('violet', [150,500], np.array([4,0]), 100))
    balls.append(Ball('yellow', [700,500], np.array([0,0]), 100))
    balls.append(Ball('firebrick', [800,500], np.array([0,0]), 100))
    balls.append(Ball('lightblue', [400,500], np.array([0,0]), 100))
elif initialState == 2:
    displayColtime = True
    mass = True
    balls.append(Ball('red', [300,350], np.array([4,0]), 100))
    balls.append(Ball('green', [700,350], np.array([-4,0]), 100))
    balls.append(Ball('blue', [500,350], np.array([0,0]), 100))
    balls.append(Ball('violet', [500,550], np.array([0,-4]), 100))
    balls.append(Ball('yellow', [500,150], np.array([0,4]), 100))
#    balls.append(Ball('firebrick', [800,200], np.array([-2,0]), 100))
elif initialState == 3:
    collision = True
    displayColtime = False
    mass = False
#    targetFrametime = 0.1
    balls.append(Polygon(np.array([3,3]), 0.01, 'blue', 50,50,250,50,250,250,50,250))
    balls.append(Polygon(np.array([0,1]), 0.1, 'green', 500,200,600,200,600,300,500,300))
    
    
noObjects = len(balls)

one = np.zeros((noObjects,noObjects))
colTimeMat = np.zeros((noObjects,noObjects)) # collision time matrix: stores the number of frames until collision (float)
# Only populated in the lower triangle
# The index to ball association is the same as in the list 'balls'
#borderColTimeMat = np.zeros((noObjects,4))
colCorners = np.zeros((noObjects,4)) # which corner collides
borderOne = np.ones((noObjects,4))


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

#if displayColtime:
#    colDisplay = fenster.create_text(200, 100, text="initialState")

            
colPro = 0 #collision processing counter
# preprocessing
if collision :
    for i in range(len(balls)):
#        (borderColTimeMat[i],colCorners[i]) = borderColTime(i)
#        print(borderColTimeMat)
        for j in range(i):
            colTimeMat[i,j] = colTime(i,j)
            one[i,j] = 1

while frameCounter <= frame_upperBound:
    if collision : # collision is a boolean
        colList = updateColList()
        while len(colList) > 0:
#            print(colList)
            colPro += 1
            c = colList[0]
            (i,j) = (c[0], c[1])
            t = val((i,j))
            colList.remove(c)
            if 0 <= t < 1 :
                balls[i].move(i,t)
                toMove[i] = 1 - t
                if j >= 0:
                    balls[j].move(j,t)
                    toMove[j] = 1 - t
                col(i,j)
                colCounter += 1
                recomputeColTime(i, t)
                if j >= 0:
                    recomputeColTime(j, t)
                colList = updateColList()
            
    for i in range(len(balls)):
        balls[i].move(i,toMove[i])

    rec.write("ef\n")

    toMove = [1] * noObjects
    colTimeMat = colTimeMat - one
#    borderColTimeMat = borderColTimeMat - borderOne
#    if displayColtime:
#        fenster.itemconfigure(colDisplay, text=str(colTimeMat[1,0]))
#    tk.update()
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
    
#    if 0 <= computeTime < targetFrametime:
#        sleepTime = targetFrametime - computeTime
#        time.sleep(sleepTime)
    
    t3 = time.time() - t0
    avgFPS = frameCounter / t3 # running average
    
rec.close()
print("completed\n"+str(frameCounter)+ " frames were computed in " + str(t3)[:math.floor(math.log10(t3))+2] + " seconds")

#play()
