import matplotlib
import numpy
import os
import random

class Swarm():

    def __init(self):
        self.variateWeight
        self.randomLevel
        self.rows
        self.cols
        self.nr
        self.ob
        self.maxIter
        self.gridEnv
        self.robotsInit
        self.BWlist
        self.A
        self.robotBinary
        self.arrayofElements
        self.connectedRobotstRegions
        self.success
        self.binrayRobotRegions
        self.maxCellsAss
        self.minCellsAss
        self.elapsedTime
        self.discr
        self.canceled
        self.useImportance


    @staticmethod
    def gaussian2d(gaussian, x, y):
        # type: (Gaussian, float, float) -> float
        x_part = math.pow(x - gaussian.x_c, 2) / (2 * math.pow(gaussian.sigma_x, 2))
        y_part = math.pow(y - gaussian.y_c, 2) / (2 * math.pow(gaussian.sigma_y, 2))
        return gaussian.a * math.exp(-(x_part + y_part)) + 0.1

    def density_callback(self, msg):
        # type: (Gaussian) -> None
        try:
            self.gaussian = msg
            self.update_density_dist()
        except Exception as e:
            rospy.logerr("Error while getting density info " + str(e))

    def publish_voronoi(self):
        voro_tess = VoronoiTesselation()
        voro_tess.width = self.graph.width
        voro_tess.height = self.graph.height
        voro_tess.data = np.empty((voro_tess.width * voro_tess.height), dtype=int)
        for i in range(0, self.graph.width):
            for j in range(0, self.graph.height):
                voro_tess.data[i * voro_tess.width + j] = self.graph.nodes[i, j].robot_id
        # self.voronoi_publisher.publish(voro_tess)

    def tesselation_and_control_computation(self, list_robots=None):
        begin = rospy.Time.now()
        self.semaphore.acquire()
        if list_robots is None:
            list_robots = []

        for robot in self.robots.values():  # type: Robot

            pose = robot.get_pose_array()
            node = self.graph.get_node(pose)  # type: Node
            node.cost = 0  # np.linalg.norm(np.subtract(node.pose, robot.get_pose_array()))
            node.power_dist = node.cost - pow(robot.weight, 2)
            robot.control.control_law.clear_i()
            # robot.mass = self.get_density(node)*math.pow(self.graph.resolution, 2)
            self.priority_queue.put((node.power_dist, node, robot.id))
            self.mark_node(node, robot)
            for q in node.neighbors:  # type: Node
                if q is not node and not bool(set(q.obstacle_neighbors) & set(node.obstacle_neighbors)):
                    q.s = q

        h_func = 0
        iterations = 0

        while not self.priority_queue.empty():
            iterations = iterations + 1
            elem = self.priority_queue.get()
            q = elem[1]  # type: Node
            if q.power_dist == float('inf'):
                break
            if q.robot_id is not -1:
                continue

            q.robot_id = elem[2]
            robot = self.robots[elem[2]]  # type: Robot
            robot_node = self.graph.get_node(robot.get_pose_array())  # type: Node

            h_func = h_func + (pow(q.power_dist, 2) + pow(robot.weight, 2)) * self.density[
                q.indexes[0], q.indexes[1]] * pow(self.graph.resolution, 2)
            self.mark_node(q, robot)

            if q.s is not None:
                i_cl = self.get_density(q) * q.cost * np.subtract(q.s.pose, robot.get_pose_array())
                robot.control.control_law.add_control_law(i_cl)

            for n in q.neighbors:  # type: Node
                _cost = q.cost + np.linalg.norm(np.subtract(q.pose, n.pose))
                _power_dist = self.power_dist(_cost, robot.weight)
                if _power_dist < n.power_dist:
                    n.cost = _cost
                    n.power_dist = _power_dist
                    if not n.is_neighbor(robot_node):
                        n.s = q.s
                    self.priority_queue.put((n.power_dist, n, robot.id))
                else:
                    if n.robot_id is not -1:
                        robot.neighbors[n.robot_id] = self.robots[n.robot_id]
                        self.robots[n.robot_id].neighbors[robot.id] = robot

        for robot in self.robots.values():  # type: Robot
            if robot.id in list_robots:
                # print("\n\nRobot " + str(robot.id))
                control_integral = robot.control.control_law.get_control_integral()
                # print("Control integral: " + str(control_integral))
                robot_node = self.graph.get_node(robot.get_pose_array())
                # self.mark_node(robot_node, self.robot_color)
                best_node = self.get_best_aligned_node(control_integral, robot_node)  # type: Node
                if best_node is None:
                    rospy.logerr("Best node is none robot_" + str(robot.id))
                    node_l = self.get_largest_weight_neighbor(robot_node)
                    if node_l is not None:
                        robot.control.set_goal(node_l.pose)
                else:
                    robot.control.set_goal(best_node.pose)

        self.publish_tesselation_image()
        self.adapt_weights()
        self.clear()
        time_diff = (rospy.Time.now() - begin).to_sec()
        rospy.loginfo("Finished! iter=" + str(iterations) + ",h = " + str(h_func) + ", " + str(time_diff) + "s")
        self.semaphore.release()
        self.h_file.write("{0} {1}\n".format(str(h_func), str((rospy.Time.now() - self.time_begin).to_sec())))
        self.h_file.flush()
        return h_func

    def constructAssignmentM():

    def calculateRobotBinaryArrays(self):

    def finalUpdateOnMetricMatrix(self):

    def generateRandomMetrix(self):

    def calculateCriterionMatrix():

    def isThisAGoalState(self):

    def calcConnectedMultiplier(self):

    def assign(self):

    def EuclideanDis(self):

    def EuclideanInt(self):

    def deepCopyMatrix(self):

    def deepCopyListMatrix(self):

    def deepCopyMatrixInt(self):

    def defineRobotsObstacles(self):

    def printMatrix(self):

    def getSuccess(self):

    def getNr(self):

    def getNumOB(self):

    def getAssignmentMatrix(self):

    def getRobotBinary(self):


