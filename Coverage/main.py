import matplotlib
import numpy as np
import os
import random
import time

class Swarm():


    def __init__(self, id_r, weight, color, xd=0, yd=0):
        self.id = id_r
        self.weight = weight
        self.color = color
        self.speed_v = 0
        self.speed_w = 0
        self.speed_pub = None
        self.pose_sub = None
        self.pose = None
        self.first_pose = None
        self.mass = 0
        self.xd = xd
        self.yd = yd
        self.neighbors = {}

    def __init(self, r, c, src, iters, vWeight, rLevel, discr, imp):
        self.variateWeight = vWeight
        self.randomLevel = rLevel
        self.rows = r
        self.cols = c
        self.nr = 0
        self.ob = 0
        self.maxIter = iters
        self.gridEnv = deepCopyMatrix(src)
        self.robotsInit = []
        self.BWlist
        self.A = np.zeros((r, c))
        self.robotBinary = (np.zeros((r, c)) > 0)
        self.arrayofElements
        self.connectedRobotstRegions
        self.success
        self.binrayRobotRegions
        self.maxCellsAss
        self.minCellsAss
        self.elapsedTime
        self.discr = discr
        self.canceled = False
        self.useImportance = imp

    @staticmethod
    def constructAssignmentM(self):
        startTime = time.time()
        NoTiles = self.rows*self.cols
        dawd