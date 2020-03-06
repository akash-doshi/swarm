import numpy as np
from numpy.random import default_rng
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
import random
from shapely.ops import polygonize,unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
import geovoronoi
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
#https://ipython-books.github.io/145-computing-the-voronoi-diagram-of-a-set-of-points/


def random_color(as_str=True, alpha=0.5):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(np.array(rgb)/255) + [alpha]


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a
    2D diagram to finite regions.
    Source:
    [https://stackoverflow.com/a/20678647/1595060](https://stackoverflow.com/a/20678647/1595060)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a
    # given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))
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
            # Compute the missing endpoint of an
            # infinite ridge
            t = vor.points[p2] - \
                vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]]. \
                mean(axis=0)
            direction = np.sign(
                np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + \
                direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v]
                         for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[
            np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def plot_polygons(polygons, ax=None, alpha=0.5, linewidth=0.7, saveas=None, show=True):
    # Configure plot
    if ax is None:
        plt.figure(figsize=(5,5), frameon=False)
        ax = plt.subplot(111)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.axis("equal")
    # Set limits
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    # Add polygons
    for poly in polygons:
        colored_cell = Polygon(poly,
                               linewidth=linewidth,
                               alpha=alpha,
                               facecolor=random_color(as_str=False, alpha=1),
                               edgecolor="black")
        ax.add_patch(colored_cell)
    if not saveas is None:
        plt.savefig(saveas)
    if show:
        plt.show()
    return ax


def main():

    df = pd.read_csv('data.csv', header=None)
    lon = df[1]
    lat = df[2]
    #points = np.delete(df.values, 0, 1)
    #print(points)
    box = (0, 0,
           15, 15)

    area = [[0,0], [15,0], [15,13], [13,13], [13,15], [0,15]]

    ext = [(0, 0), (15, 0), (15, 13), (13, 13), (13, 15), (0, 15)]
    int = [(11, 12), (12, 12), (12, 11), (11, 11)]
    grint = [(1, 14), (1, 12), (3, 12), (3, 14)]

    area_shape = Polygon(ext, [grint, int])

    coords = np.random.randint(1, 12, size=(40, 2))
    print(coords)

    points = []
    for point in coords:
        if area_shape.contains(Point(point)) is True:
            points.append(point)

    points = coords
    #points = [point for point in coords if area_shape.contains(Point(point)) is True]
    print("points:")

    print(area_shape)
    #area_shape = area.iloc[0].geometry

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    area = world[world.name == 'Italy']

    area = area.to_crs(epsg=3395)  # convert to World Mercator CRS
    #area_shape = area.iloc[0].geometry  # get the Polygon
    print(area_shape)

    vor = spatial.Voronoi(points)

    #regions, vertices = voronoi_finite_polygons_2d(vor)

    poly_shapes, pts, poly_to_pt_assignment = voronoi_regions_from_coords(points, area_shape)

    print(dir(poly_shapes))
    print(dir(pts))
    print(poly_to_pt_assignment)
    fig, ax = subplot_for_map()
    plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, points, poly_to_pt_assignment)

    polygons = []
    #for reg in regions:
    #    polygon = vertices[reg]
    #    polygons.append(polygon)

    #plot_polygons(polygons)

    #pts = MultiPoint([Point(i) for i in points])
    #mask = pts.convex_hull
    #new_vertices = []
    #for region in regions:
    #    polygon = vertices[region]
    #    shape = list(polygon.shape)
    #    shape[0] += 1
    #    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    #    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    #    new_vertices.append(poly)
    #    plt.fill(*zip(*poly), alpha=0.4)
    #plt.plot((points)[:, 0], (points)[:, 1], 'ko')
    #plt.title("Clipped Voronois")
    #plt.show()

    # We generate colors for districts using a color map.

#    ax = m.show_mpl(figsize=(12, 8))
#    ax.add_collection(
#        mpl.collections.PolyCollection(
#            cells, facecolors=colors,
#            edgecolors='k', alpha=.35))
#
    plt.show()


if __name__ == '__main__':
    main()
