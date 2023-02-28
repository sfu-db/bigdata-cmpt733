"""Utilities for graph analytics assignment - graph tool and linear algebra
"""

import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

import graph_tool.all as gt
assert f"{gt.__version__}" >= "2.37"

def v_id(name, g, v_name):
    """Return integer index of vertex with given `name`
    Args:
        name - name of vertex (str)
        g - graph tool graph
        v_name - vertex property array giving names for each vertex
                 as returned when calling g.add_edge_list with hashed=True
    """
    # obtain vertex object
    v = gt.find_vertex(g, v_name, name)[0]
    # get int index
    return g.vertex_index[v]

def edge_vector(name_from, name_to, pos, v_name):
    """Determine 2d-vector between positions of two named nodes (aka vertices)
    Args:
        name_from, name_to - names of vertices
        pos - positional vertex property as computed by graph tool layout algorithm
        v_name - vertex property array giving names for each vertex
                 as returned when calling g.add_edge_list with hashed=True
    Returns:
        2-dimensional vector of positional difference between the two nodes
    """
    g = pos.get_graph()
    evec = (pos[v_id(name_to, g, v_name)].a -
            pos[v_id(name_from, g, v_name)].a)
    return evec

def rot2d_rad(angle):
    """Make a 2D rotation matrix for `angle` given in radians. """
    return (R.from_rotvec(angle * np.array([0, 0, 1])) # rotate around out-of-plane z-axis
             .as_matrix()[:2,:2])                      # use top left 2x2 sub matrix

def rot2d_deg(angle):
    """Make a 2D rotation matrix for `angle` given in degrees. """
    return rot2d_rad(angle*np.pi/180)

def rot2d_vec(v, v_as_x=True):
    """Make a 2D rotation matrix that rotates a canonical axis to align with vector `v`
       To align `v` with canonical axis, instead, use the inverse mat.I of the returned matrix mat.
    Args:
        v - vector to rotate to
        v_as_x - if True, rotate the x-axis (default), otherwise rotate y-axis to align with v
    Returns:
        numpy rotation matrix
    """
    v /= LA.norm(v)
    if v_as_x:
        mat = np.matrix([v, v[::-1]])
        mat[0,1] *= -1
    else:
        mat = np.matrix([v[::-1], v])
        mat[1,0] *= -1
    #assert abs(LA.det(mat)-1)<1e-4, "Rotation matrix should have unit determinant"
    return mat

def rotate_vertex_pos(pos, r):
    """Apply rotation matrix `r` to vertex positions `pos` in place."""
    pos.set_2d_array(np.dot(r, pos.get_2d_array(pos=[0,1])))

def rotate_layout(pos, name_from, name_to, v_name, align_x=True):
    """Rotate vertex positions `pos` such that two vertex nodes align with each other on x (or y) axis.
       The operation modifyies `pos` in-place.
    Args:
        pos - positional vertex property as computed by graph tool layout algorithm
        name_from, name_to - names of the two nodes to align along an axis
        v_name - vertex property array giving names for each vertex
                 as returned when calling g.add_edge_list with hashed=True
        align_x - if True, rotate vertices into x-axis (default), otherwise rotate into y-axis
    """
    rotation_matrix = rot2d_vec(edge_vector(name_from, name_to,
                                pos, v_name),
                                v_as_x=align_x).I
    rotate_vertex_pos(pos, rotation_matrix)
