# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:16:49 2019

@author: Clément
"""
import time, math
from tkinter import Tk, Canvas

#canvas size
WIDTH = 1000
HEIGHT = 700

fileName ="recBall1.txt" #Name of the file to be read and played

targetFPS = 61
targetFrametime = 1 / targetFPS

dt = True #display time

def readColor(s):
    for i in range(len(s)):
        if s[i] == " ":
            return [s[0:i],s[i+1:]]

def readNumbers(s): # turns s into a list of floats
    i = 0
    p = []
    while s != '\n':
        if s[i] == " ":
#            print(s[0:i])
            p.append(float(s[0:i]))
            s = s[i+1:]
            i = 0
        else:
            i += 1
    return p


def play():
    tk = Tk()
    fenster = Canvas(tk, width=WIDTH, height=HEIGHT, bg="white")
    tk.title("The night is dark and full of terrors")#"When you play the game of thrones, you win or you die")
    fenster.pack()
    if dt:
        timeDisplay = fenster.create_text(WIDTH - 20, HEIGHT - 10, text="initialState")
                    
    file = open(fileName,'r')
    commandList = file.readlines()
    
    objects = []
    colors = []
    
    frameCounter = 0
    t0 = time.time()
    
    
    for c in commandList:    
        if c[0:2] == "mp":
            p = readNumbers(c[3:])
            i = int(p[0])
            fenster.delete(objects[i])
            objects[i] = fenster.create_polygon(p[1:],fill=colors[i])
        elif c[0:2] == "mb":
            p = readNumbers(c[3:])
            i = int(p[0])
            fenster.delete(objects[i])
            objects[i] = fenster.create_oval(p[1:],fill=colors[i])
        elif c[0:2]== "ef":
            frameCounter += 1
            if dt:
                fenster.itemconfigure(timeDisplay, text=str(frameCounter * targetFrametime)[0:5])

            if time.time() - t0 < frameCounter * targetFrametime:
                time.sleep(frameCounter * targetFrametime - (time.time()- t0))
            tk.update()
        elif c[0:2]== "cp":
            (color,c) = readColor(c[3:])
            p = readNumbers(c)
            polygon = fenster.create_polygon(p,fill=color)
            objects.append(polygon)
            colors.append(color)
        elif c[0:2] == "cb":
            (color,c) = readColor(c[3:])
            p = readNumbers(c)
            b = fenster.create_oval(p,fill=color)
            objects.append(b)
            colors.append(color)
    
    file.close()
    
    while True :
        tk.update()
    
#play()
