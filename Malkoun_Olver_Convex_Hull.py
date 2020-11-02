#!/usr/bin/env python
# coding: utf-8

# Import the necessary modules/libraries
# import autograd.numpy as np
import numpy as np
from autograd import grad, jacobian
# import numdifftools as nd
import sympy as sp
from numpy import poly1d
from scipy import linalg
import math
import random
import itertools
from sympy import symarray
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
from tkinter import *
from tkinter.colorchooser import askcolor
from scipy.spatial import ConvexHull
from scipy.integrate import quad, dblquad, fixed_quad
from scipy.special import p_roots
from time import perf_counter
import datetime
import PIL.Image

def sigmoid(x):
    return np.exp(x)/(1.+np.exp(x))

# define map_ij from S^{d-1} to R^d associated to the direction v_ij
def map_ij(v,v_ij,option, eps = 0.1):
    if len(v.shape) == 1:
        v = v.reshape(1,-1)
    if option == 'chordal':
        return np.sqrt(np.sum((v - v_ij)**2,axis=1))
    elif option == 'spherical':
        return np.arccos(np.dot(v,v_ij))**2
    elif option == 'Cauchy-Schwarz':
        return 0.5*(np.ones(v.shape[:-1],dtype=np.float)-np.dot(v,v_ij))
    elif option == 'Malkoun-Olver':
        return eps*np.ones(v.shape[:-1],dtype=np.float)             +np.max(np.array([[-np.dot(v,v_ij)],                            [np.zeros(v.shape[:-1],dtype=np.float)]]),axis=0)

# given an np array with the n points in R^d as rows, compute the associated n maps
# map_i from S^{d-1} to R^d
def mycoeffs(x,v,option, eps = 0.1,full_output=False):
    n=x.shape[0]
    d=x.shape[1]
    v_shape = v.shape[:-1]
    if v_shape == ():
        v_shape = (1,)
    vecs=np.zeros((n,n,d),dtype=np.float)
    output = np.ones((n,n)+v_shape, dtype = np.float)
    
    for i in range(n-1):
        for j in range(i+1,n):
            vec_ij = x[j,:]-x[i,:]
            vec_ij = vec_ij/np.linalg.norm(vec_ij)
            output[i,j] = map_ij(v,vec_ij,option,eps)
            output[j,i] = map_ij(v,-vec_ij,option,eps)
            
    if (option == 'chordal') or (option == 'spherical') or             (option == 'Cauchy-Schwarz'):
        return np.prod(output, axis = 1)
    elif (option == 'Malkoun-Olver'):
        if not full_output:
            return np.prod(output, axis = 1)    
        else:
            return np.prod(output, axis = 1), output
        
# returns a pseudo-random number between -a and a
def rand(a):
    return 2*a*random.random()-a

# generate N configurations of n points having each coordinate
# between -a and +a, put these N configurations in an x with 3
# indices, such that the distances between 0 and x_i are all greater or equal to a*eps
# and the distances between x_i and x_j, and those between x_i and -x_j are all
# greater or equal to a*eps. compute the corresponding N determinants and put them in
# an np array called MyD
# return x and MyD.
# eps should be small enough so that there are configurations with those requirements, but not
# too small, otherwise one would get rounding errors
# as a rule of thumb, take a to be approximately between 1.5*n and 2*n, eps to be 0.1 for N <= 30
def test(d,n,N,a = 6.,eps = 0.1):
    x=np.zeros((N,n,d),dtype=np.float)
    
    for k in range(N):
        for i in range(n):
            condition = True
            while condition:
                for j in range(d):
                    x[k,i,j]=rand(a)
                condition = (linalg.norm(x[k,i,:])<a*eps)
                for l in range(i):
                    condition = condition | (linalg.norm(x[k,i,:]-x[k,l,:])<a*eps) | (linalg.norm(x[k,i,:]+x[k,l,:])<a*eps)
#                 elif planar:
#                     for j in range(2):
#                         x[k,i,j]=rand(a)
#                     condition = (linalg.norm(x[k,i,:])<a*eps)
#                     for l in range(i):
#                         condition = condition | (linalg.norm(x[k,i,:]-x[k,l,:])<a*eps) | (linalg.norm(x[k,i,:]+x[k,l,:])<a*eps)
    
    return x

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'red'
    PALETTE = ['spring green', 'gold', 'deep sky blue', 'orange', 'cornflower blue']
    N_COLORS = len(PALETTE)
    m=60
    eps=0.01
    counter = 1
    timestamp = '_{:%Y-%m-%d %H:%M:%S}_'.format(datetime.datetime.now())
    OPTIONS = ['chordal', 'spherical', 'Cauchy-Schwarz', 'Malkoun-Olver']
    
    def __init__(self):
        self.root = Tk()
        self.root.title("Convex Hull Experiments")
        self.variable = StringVar(self.root)
        self.variable.set(self.OPTIONS[3]) # default value
        self.infovar = StringVar(self.root)
        self.infovar.set("Important Information will be displayed here.")

#         self.pen_button = Button(self.root, text='Add Pt', command=self.use_pen)
#         self.pen_button.grid(row=0, column=0)

        self.del_pt_button = Button(self.root, text='Del Pt', command=self.del_pt)
        self.del_pt_button.grid(row=0, column=0)

        self.draw_button = Button(self.root, text='Draw Curve', command=self.draw)
        self.draw_button.grid(row=0, column=1)
        
        self.del_curve_button = Button(self.root, text='Del Curve', command=self.del_curve)
        self.del_curve_button.grid(row=0, column=2)

        self.reset_button = Button(self.root, text='Reset', command=self.reset)
        self.reset_button.grid(row=0, column=3)
        
        self.save_button = Button(self.root, text='Save', command=self.save)
        self.save_button.grid(row=0, column=4)
        
        self.menu = OptionMenu(self.root,self.variable,*self.OPTIONS)
        self.menu.grid(row=1,column=5,columnspan=2)
        
        self.m_label = Label(self.root, text="m = ")
        self.m_label.grid(row=2, column=5)
        
        self.m_text = Text(self.root, height=1, width=3)
        self.m_text.insert('1.0', str(self.m))
        self.m_text.grid(row=2, column=6)
        
        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, column=0, rowspan=3, columnspan=5)
        
        self.eps_label = Label(self.root, text="eps = ")
        self.eps_label.grid(row=3, column=5)
        
        self.eps_text = Text(self.root, height=1, width=10)
        self.eps_text.insert('1.0', str(self.eps))
        self.eps_text.grid(row=3, column=6)
        
        self.info = Label(self.root, textvariable=self.infovar)
        self.info.grid(row=4,columnspan=5)
        
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.color_index = 0
        self.points = []
        self.ovals = []
        self.curves = []
        self.line_width = 5 # self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.draw_button
        self.c.bind('<Button-1>', self.add_point)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def del_pt(self):
        self.activate_button(self.del_pt_button)
        if self.points != []:
            self.c.delete(self.ovals[-1])
            self.ovals.pop()
            self.points.pop()

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def add_point(self, event):
        self.line_width = 5 # self.choose_size_button.get()
        paint_color = self.PALETTE[self.color_index]
        self.points.append([event.x,event.y])
        self.ovals.append(self.c.create_oval(event.x-self.line_width, event.y-self.line_width,                           event.x+self.line_width, event.y+self.line_width,                           fill = paint_color))


    def draw(self):
        self.infovar.set("Important Information will be displayed here.")
        self.small_line_width = 1
        self.m = int(self.m_text.get('1.0',END))
        self.eps = np.float(self.eps_text.get('1.0',END))
        n = len(self.points)
        d = 2
        myconf = np.array(self.points)
        option = self.variable.get()
        
        if option == 'Malkoun-Olver':
            den = lambda v: np.sum(mycoeffs(myconf,v,option,self.eps),axis=0)                        .reshape(-1,1)
            num = lambda v: mycoeffs(myconf,v,option,self.eps).T.dot(myconf)
        else:
            den = lambda v: np.sum(mycoeffs(myconf,v,option,self.eps)**(self.m),axis=0).reshape(-1,1)
            num = lambda v: (mycoeffs(myconf,v,option,self.eps)**(self.m)).T.dot(myconf)
        def geometric_map(v):
            if den(v).any() == 0.:
                ntimes = np.sum((den(v) == 0.), axis = 0)
                self.infovar.set("Warning! Division by 0 encountered "                       + str(ntimes) + "times!")
                return
            else:
                return num(v)/den(v)
        mycurve = []

        for r in [1.]:
            circle = r*np.exp(1.j*np.linspace(0.,2.*np.pi,50000, endpoint=True))
            pts_on_circle = np.array([circle.real, circle.imag]).T
            curve = geometric_map(pts_on_circle)

            for ind in range(len(curve)):
                mycurve.append(self.c.create_oval(curve[ind][0]-self.small_line_width,                                curve[ind][1]-self.small_line_width,                                curve[ind][0]+self.small_line_width,                                curve[ind][1]+self.small_line_width,
                                fill='red', outline='red'))
                              
        self.color_index = (self.color_index + 1) % self.N_COLORS

        self.color_index = (self.color_index + 1) % self.N_COLORS
        self.curves.append(mycurve)
    
    def del_curve(self):
        if self.curves != []:
            mycurve = self.curves.pop()
            for line in mycurve:
                self.c.delete(line)
    
    def reset(self):
        self.c.delete("all")    
        self.ovals = [] 
        self.points = []
        self.curves = []
        self.color_index = 0
        
    def save(self):
        PATH = "/Users/joey/Desktop/"
        self.c.postscript(file=PATH+"Figure"+self.timestamp+str(self.counter)+".eps")
        self.counter += 1

if __name__ == '__main__':
    Paint()

