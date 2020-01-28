# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 12:39:09 2018

@author: Vivien
"""

import pandas as pd
import numpy as np
import scipy.spatial
import pyproj
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon

data_path = 'C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Processed Water Level Data\\'

#pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth',100)
sensor_meta = pd.read_table(data_path+'wl_position_meta.csv',sep=',',index_col=False)

xy_pos = sensor_meta[['sensor','latitude','longitude']].drop_duplicates().reset_index(drop = True)
xy_pos = xy_pos[xy_pos.sensor.str.contains("WLW")]
#xy_pos = xy_pos[xy_pos.sensor.str.contains("12") == False]
#xy_pos = xy_pos[xy_pos.sensor.str.contains("13") == False]
#xy_pos = xy_pos[xy_pos.sensor.str.contains("14") == False]
xy_arr = np.array(xy_pos[['longitude','latitude']])

p1 = pyproj.Proj(init='epsg:4326')
p2 = pyproj.Proj(init='epsg:3443')
x0,y0 = p2(-87.693463,41.604446)
xn,yn = p2(-87.681585,41.608161)
xn = xn - x0
yn = yn - y0

prairie_poly = np.array([[0,0], [0,yn], [xn,yn], [xn,0]])

xy_m = []
for item in xy_arr:
    x1 = item[0]
    y1 = item[1]
    x2, y2 = p2(x1,y1)
    x2 = x2 - x0
    y2 = y2 - y0
    xy_m = xy_m + [[x2,y2]]

# Add dummy points well outside the bounds so that the "edge" regions are closed and areas can be calculated
#xy_m = xy_m + [[-500,500],[-500,-500],[1500,500]]
#xy_pos = 

#(https://gist.github.com/pv)    
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# Scipy function for creating the voronoi polygons    
vor = scipy.spatial.Voronoi(xy_m,qhull_options='Qbb Qc Qx')
# Code off github (function above) for determining locations of distant points so that areas can be calculated
vor.regions, vor.vertices =  voronoi_finite_polygons_2d(vor)

prairie_box = box(0,0,xn,yn) # Set up bounds of prairie
vertices_coords = []
enclosed_areas = []
for region in vor.regions:
    vc = []
    for vertex in region:
        vc = vc + [vor.vertices[vertex].tolist()]
    
    vertices_coords = vertices_coords + [vc]
    ea = Polygon(vc).intersection(prairie_box).area
    aa = Polygon(vc).area
    enclosed_areas = enclosed_areas + [ea]
xy_pos['coord_m'] = xy_m    
xy_pos['vertices_coords'] = vertices_coords
xy_pos['encl_area_m'] = enclosed_areas

fig = scipy.spatial.voronoi_plot_2d(vor)
plt.axis("equal")
fig = scipy.spatial.voronoi_plot_2d(vor)
plt.axis([0,xn,0,yn])


for poly in vertices_coords:
    plt.fill(*zip(*poly), alpha=0.4)