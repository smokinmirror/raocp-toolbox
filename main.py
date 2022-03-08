import raocp as r
import numpy as np
import turtle


def circle_coord(rad, arc):
    return rad * np.cos(np.deg2rad(arc)), rad * np.sin(np.deg2rad(arc))


def goto_circle_coord(trt, rad, arc):
    trt.penup()
    trt.goto(circle_coord(rad, arc))
    trt.pendown()


def draw_circle(trt, rad):
    trt.penup()
    trt.home()
    trt.goto(0, -rad)
    trt.pendown()
    trt.circle(rad)


p = np.array([[0.1, 0.8, 0.1],
              [0.4, 0.6, 0],
              [0, 0.3, 0.7]])

v = np.array([0.5, 0.4, 0.1])

(N, tau) = (10, 7)

tree = r.core.MarkovChainScenarioTreeFactory(transition_prob=p,
                                             initial_distribution=v,
                                             num_stages=N, stopping_time=tau).create()
print(tree.nodes_at_stage(2))
tree.bulls_eye_plot()



