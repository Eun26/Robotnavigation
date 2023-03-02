#!/usr/bin/python
import math
import sys
import time
import pickle
import numpy as np
import random
import cv2

from itertools import product
from math import cos, sin, pi, sqrt

from plotting_utils import draw_plan_rrt
from priority_queue import priority_dict


class State(object):
    """
    2D state. 
    """

    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []

    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))

    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x) ** 2 + (state.y - self.y) ** 2)


class RRTPlanner(object):
    """
    Applies the RRT algorithm on a given grid world
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:, :, 0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby 
        surroundings are free.
        """
        return (self.occ_grid[state.y - 5:state.y + 5, state.x - 5:state.x + 5] == 0).all()

    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image. 
        """
        # TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}

        ## 모든 값은 전부 int 형이어야 한다.
        ## random.randint 를 통해 world의 전체 크기를 벗어나지 않도록 x, y를 설정해준다.
        x = random.randint(0, self.occ_grid.shape[0] - 1)
        y = random.randint(0, self.occ_grid.shape[1] - 1)

        return State(x, y, None)

    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        curr_ptr = state
        path = [state]

        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        # return a reverse copy of the path (so that first state is starting state)
        return path[::-1]

    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state

    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:

        
        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y

        
        Otherwise, s_rand is farther than max_radius from s_nearest. 
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.
        
        """
        dx = s_rand.x - s_nearest.x  # 아래에서 비율에 따라 x, y 값을 설정해주기 위해 dx, dy 를 구함.
        dy = s_rand.y - s_nearest.y
        d = sqrt(dx ** 2 + dy ** 2)  # 아래에서 직선의 비율에 맞게 x, y를 설정해주기 위하여 두 state사이의 거리 d 구함.
        print(dx, dy, d)

        # TODO: populate x and y properly according to the description above.
        # Note: x and y are integers and they should be in {0, ..., cols -1}
        # and {0, ..., rows -1} respectively

        if d <= max_radius:  # 만약 max_radius보다 d가 작다면, 그대로 x, y를 가져온다.
            x = s_rand.x
            y = s_rand.y

        elif d > max_radius:  # 아니라면, s_nearest와 s_rand가 이루는 직선의 기울기를 고려한 x, y값을 새로이 설정해준다. (max_radius초과하지 않게!)
            x = s_nearest.x + int(dx / d * max_radius)
            y = s_nearest.y + int(dy / d * max_radius)

        s_new = State(x, y, s_nearest)
        # print("after steer towards s_new" )
        # print(s_new.x ,s_new.y)
        return s_new

    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))
        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10

        dx = s_to.x- s_from.x   ##두 state간의 직선의 기울기에 맞추어 collision검사를 하기위해 dx, dy, dist를 구하였다.
        dy = s_to.y- s_from.y
        dist = sqrt(dx ** 2 + dy ** 2)
        #"""
        print(dist)
        for i in xrange(0, max_checks):
            # TODO: check if the interpolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False
            x = s_to.x - int(dx * float(i)/float(max_checks))   ## x는 처음에 s_to와 s_from사이의 직선을 긋고 점점 짧게 하여 collision확인.
            y = s_to.y - int(dy * float(i)/float(max_checks))   ## y 도 마찬가지 이다.
            print(self.state_is_free(State(x, y, s_from)))      ## 위에서 쓰인 함수 state_is_free를 통해 state주변의 장애물을 확인할 수 있다.
            if self.state_is_free(State(x, y, s_from)):         ## state_is_free가 true이면 장애물이 없다는 얘기이므로, return으로 true를 넣어준다.
                if i == max_checks-1 :
                    return True
                continue
            else :
                return False                                   ##하나라도 false가 나오면 return으로 false를 주고, break를 해준다.
                break


    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # The set containing the nodes of the tree
        tree_nodes = set()
        tree_nodes.add(start_state)

        # image to be used to display the tree
        img = np.copy(self.world)

        plan = [start_state]

        for step in xrange(max_num_steps):

            # TODO: Use the methods of this class as in the slides to
            # compute s_new
            s_rand = self.sample_state()     # 처음 점은 random으로 주어준다.
            print("   s_rand   ")
            print(s_rand.x, s_rand.y)
            s_nearest = self.find_closest_state(tree_nodes, s_rand)   # tree_nodes에 저장 된 점 중에서 s_rand와 가장 가까운 점 선택.
            print("   s_nearest   ")
            print(s_nearest.x, s_nearest.y)
            s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)   # max_radius를 넘지 않는 새로운 state를 뽑아준다.
            print("   s_new   ")
            print(s_new.x, s_new.y)

            if self.path_is_obstacle_free(s_nearest, s_new):
                print("new node is included into ..")    # collision이 없다면, s_new를 tree_nodes에 넣어준다.
                tree_nodes.add(s_new)
                s_nearest.children.append(s_new)

                # If we approach the destination within a few pixels
                # we're done. Return the path.
                if s_new.euclidean_distance(dest_state) < dest_reached_radius:
                    dest_state.parent = s_new
                    plan = self._follow_parent_pointers(dest_state)
                    break

                # TODO:plot the new node and edge
                #cv2.circle(img, (s_new.x, s_new.y), int(math.hypot(s_new.x, s_new.y)), (255,0,0))
                cv2.circle(img, (s_new.x, s_new.y), 1, (255, 0, 0))        # s_new에 해당하는 점을 circle로 만들어준다.
                cv2.line(img, (s_new.x, s_new.y), (s_nearest.x, s_nearest.y), (255,0,0), 1)     ## 각 점을 잇는 line을 그어준다.

            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            cv2.imshow('image', img)
            cv2.waitKey(10)

        draw_plan_rrt(img, plan, bgr=(0, 0, 255), thickness=2)
        cv2.waitKey(0)
        return [start_state]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: rrt_planner.py occupancy_grid.pkl"
        sys.exit(1)

    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    rrt = RRTPlanner(world)

    start_state = State(10, 10, None)
    dest_state = State(500, 500, None)

    max_num_steps = 1000  # max number of nodes to be added to the tree
    max_steering_radius = 30  # pixels
    dest_reached_radius = 50  # pixels
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)
