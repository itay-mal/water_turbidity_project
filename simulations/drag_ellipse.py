import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backend_bases import MouseEvent
import numpy as np

ELLIPSE_CENTER = (50,50)
ELLIPSE_HEIGHT = 60
ELLIPSE_WIDTH = 80

class DraggablePlotExample(object):
    u""" An example of plot with draggable markers """

    def __init__(self):
        self._figure, self._axes, self._ellipse = None, None, None
        self._ellipse_center, self._ellipse_top, self._ellipse_right = None, None, None
        self._dragging_point = None
        self._points = {}

        self._init_plot()

    def _init_plot(self):
        self._figure = plt.figure("Example plot")
        axes = plt.subplot(1, 1, 1)
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)
        axes.grid(which="both")
        self._axes = axes
        self._ellipse_center = ELLIPSE_CENTER
        self._ellipse_top   = (self._ellipse_center[0], self._ellipse_center[1] + int(ELLIPSE_HEIGHT/2))
        self._ellipse_right = (self._ellipse_center[1] + int(ELLIPSE_WIDTH/2), self._ellipse_center[1])

        # add initial calculation of ellipse and its anchors
        self._points = (self._ellipse_center,
                        self._ellipse_top,
                        self._ellipse_right)
        self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._update_plot()
        plt.show()

    def _update_plot(self):
        # redraw ellipse
        width = 2 * (self._ellipse_right[0] - self._ellipse_center[0])
        height = 2 * (self._ellipse_top[1] - self._ellipse_center[1])
        self._ellipse = Ellipse(xy=self._ellipse_center, width=width, height=height,
                          edgecolor='r', fc='None', lw=2)
        self._axes.add_patch(self._ellipse)

        # draw markers
        self._axes.scatter(self._ellipse_center[0], self._ellipse_center[0], marker='+', c='b', linewidths=5)
        print(self._points)
        self._figure.canvas.draw()

    def _add_point(self, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = int(x.xdata), int(x.ydata)
        self._points[x] = y
        return x, y

    def _remove_point(self, x, _):
        if x in self._points:
            self._points.pop(x)

    def _find_neighbor_point(self, event):
        u""" Find point around mouse position

        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 3.0
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for x, y in self._points:
            distance = math.hypot(event.xdata - x, event.ydata - y)
            if distance < min_distance:
                min_distance = distance
                nearest_point = (x, y)
        if min_distance < distance_threshold:
            print('caught point')
            return nearest_point
        return None

    def _on_click(self, event):
        u""" callback method for mouse click event

        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            else:
                pass # self._add_point(event)
            self._update_plot()
        # TODO: do we need right-click
        # right click
        # elif event.button == 3 and event.inaxes in [self._axes]:
        #     point = self._find_neighbor_point(event)
        #     if point:
        #         self._remove_point(*point)
        #         self._update_plot()

    def _on_release(self, event):
        u""" callback method for mouse release event

        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        u""" callback method for mouse motion event

        :type event: MouseEvent
        """
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._remove_point(*self._dragging_point)
        self._dragging_point = self._add_point(event)
        self._update_plot()


if __name__ == "__main__":
    plot = DraggablePlotExample()