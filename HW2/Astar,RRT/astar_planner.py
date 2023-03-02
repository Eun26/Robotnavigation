#!/usr/bin/python
import math
import sys
import time
import pickle
import numpy as np
from itertools import product
from math import cos, sin, pi, sqrt 

from plotting_utils import draw_plan
from priority_queue import priority_dict
from priority_queue import PriorityQueue

class State(object):
    """
    2D state. 
    """
    
    def __init__(self, x, y):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y

        
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
    
    
class AStarPlanner(object):
    """
    Applies the A* shortest path algorithm on a given grid world
    """
    
    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world.sum(axis=2) / 3
        print(self.world)

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world
        self.occ_grid = (self.occ_grid  < 255).astype('uint8')
        print(self.occ_grid.max())
        
    def state_is_free(self, state):
        """
        Does collision detection. Returns true if the state and its nearby
        surroundings are free.
        """
        # check = (self.occ_grid[state.x-3:state.x+3, state.y-3:state.y+3] == 1).all()
        # print(check)
        return (self.occ_grid[state.x-1:state.x+1, state.y-1:state.y+1] == 0).all()
        
    def get_neighboring_states(self, state):
        """
        Returns free neighboring states of the given state. Returns up to 8
        neighbors (north, south, east, west, northwest, northeast, southwest, southeast)
        """
        
        x = state.x
        y = state.y
        
        rows, cols = self.world.shape[:2]

        dx = [0]
        dy = [0]
        
        if (x > 0):
            dx.append(-1)

        if (x < rows -1):
            dx.append(1)

        if (y > 0):
            dy.append(-1)

        if (y < cols -1):
            dy.append(1)

        # product() returns the cartesian product
        # yield is a python generator. Look it up.
        for delta_x, delta_y in product(dx,dy):
            if delta_x != 0 or delta_y != 0:
                ns = State(x + delta_x, y + delta_y)
                if self.state_is_free(ns):
                    yield ns 
            

    def _follow_parent_pointers(self, parents, state):
        """
        Assumes parents is a dictionary. parents[key]=value
        means that value is the parent of key. If value is None
        then key is the starting state. Returns the shortest
        path [start_state, ..., destination_state] by following the
        parent pointers.
        """
        
        assert (state in parents)
        curr_ptr = state
        shortest_path = [state]
        
        while curr_ptr is not None:
            shortest_path.append(curr_ptr)
            curr_ptr = parents[curr_ptr]

        # return a reverse copy of the path (so that first state is starting state)
        return shortest_path[::-1]

    #
    # TODO: this method currently has the implementation of Dijkstra's algorithm.
    # Modify it to implement A*. The changes should be minor.
    #

    def plan(self, start_state, dest_state):

        def heuristic(state1, state2):
            return hypot((state2.x-state1.x), (state2.y-state1.y))

        """
        Returns the shortest path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # Q is a mutable priority queue implemented as a dictionary
        Q = PriorityQueue()

        Q.push(start_state, 0.0)  # key : State / value : distance

        # Array that contains the optimal distance to come from the starting state
        dist_to_come = float("inf") * np.ones((world.shape[0], world.shape[1]))
        dist_to_come[start_state.x, start_state.y] = 0

        # Boolean array that is true iff the distance to come of a state has been
        # finalized
        evaluated = np.zeros((world.shape[0], world.shape[1]), dtype='uint8')

        # Contains key-value pairs of states where key is the parent of the value
        # in the computation of the shortest path
        parents = {start_state: None}

        while Q:
            
            # s is also removed from the priority Q with this
            s = Q.pop()

            # Assert s hasn't been evaluated before
            assert (evaluated[s.x, s.y] == 0)
            evaluated[s.x, s.y] = 1 ## list 'evaluated' checks if s hasn't been visited before
                                    ## evaluated를 통해 이미 한번 방문한 포인트인지 아닌지 확인할 수 있다.
                                    ## 만약, 방문한적 없다는 것이 assert를 통해 확인된다면, 1로 바꾸어주어 evaluate 시작.
            if s == dest_state:
                return self._follow_parent_pointers(parents, s)

            # for all free neighboring states
            for ns in self.get_neighboring_states(s):
                if evaluated[ns.x, ns.y] == 1: ## If ns is the point that already visited, continue to reduce computations
                    continue                   ## ns가 이미 방문한 포인트라면, 계산량을 줄이기 위해 continue를 사용해 통과.

                #TODO: Astar 알고리즘을 구현하세요.
                #사용 변수: dist_to_come(heuristic information)

                ## dist_to_come : start_state부터의 optimal distance를 의미.
                if(dist_to_come[ns.x, ns.y] <= dist_to_come[s.x, s.y] + 1):    ## 한 grid를 1로 잡았을 때, 시작점부터 ns(이웃좌표)까지의 거리가 시작점부터 s(현좌표)까지의 거리 + 1보다 작다면,
                    continue                                                    ## 이미 최적화 되어있는 경우이므로 업데이트 하지 않는다.

                dist_to_come[ns.x, ns.y] = dist_to_come[s.x, s.y] + 1           ## 위의 경우가 아니라면, 최적화 해야하는 경우이므로,[시작점 ~ ns(이웃좌표)까지 거리]를 [시작점 ~ s(현좌표)까지 거리+1]로 업데이트

                ## Queue에서는 총 거리값, 즉 f( f = g + h)로 비교해야 한다.
                ## f 값에 따라 priority가 정해짐.
                ## g : ns 까지의 거리 , f : ns 부터 최종 목적지까지의 heuristic한 거리 값
                f_Priority = dist_to_come[ns.x, ns.y] + heuristic(ns, dest_state)

                Q.push(ns, f_Priority)    ## Queue에 넣어줌
                                          ## ns : State, f_Priority: f_distance

                parents[ns] = s ## 앞의 노드(parent node)와의 연결고리를 만들어준다.
                                ## 이후, dest_state를 찾았을 때 start_state부터 경로를 얻을 수 있음.

        return [start_state]

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: astar_planner.py occupancy_grid.pkl")
        sys.exit(1)

    # pkl_file = open(sys.argv[1], 'rb')

    # # world is a numpy array with dimensions (rows, cols, 3 color channels)
    # world = pickle.load(pkl_file)
    # pkl_file.close()
    world = np.load(sys.argv[1])
    # k_world = np.zeros((world.shape[0], world.shape[1],3))
    # k_world[:,:,0], k_world[:,:,1], k_world[:,:,2] = world, world, world
    
    print(world.shape)
    astar = AStarPlanner(world)
    # print(world.max())

    start_state = State(275, 790)
    dest_state = State(115, 900)
    
    plan = astar.plan(start_state, dest_state)
    print(len(plan))
    # print(plan)
    draw_plan(world, plan)
    
