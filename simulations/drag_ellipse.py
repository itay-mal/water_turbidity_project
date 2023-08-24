import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backend_bases import MouseEvent
from collections import defaultdict

ELLIPSE_CENTER = (50,50)
ELLIPSE_HEIGHT = 60
ELLIPSE_WIDTH = 80

class Point():
    def __init__(self, x, y, name="MyPoint"):
        self._x    = x
        self._y    = y
        self._name = name
    
    def update_point(self, x, y):
        self._x = x
        self._y = y
        return self.get_point()

    def get_point(self):
        return (self._x, self._y)

    def __add__(self, other):
        x = self._x + other._x
        y = self._x + other._y
        return Point(x, y)
    
    def __sub__(self, other):
        x = self._x - other._x
        y = self._y - other._y
        return Point(x, y)
    
    def __repr__(self):
        return "Point {} {}".format(self.get_point(), self._name)
    
    def get_name(self):
        return self._name
    

class DraggablePlotExample(object):
    """ An example of plot with draggable markers """

    def __init__(self):
        self._figure, self._axes = None, None
        self._ellipse = None
        self._ellipse_center, self._ellipse_top, self._ellipse_right = None, None, None
        self._center_annotation, self._top_annotation, self._right_annotation = None, None, None
        self._dragging_point = None
        self._points = defaultdict(Point)

        self._init_plot()

    def _init_plot(self):
        self._figure = plt.figure("Example plot")
        axes = plt.subplot(1, 1, 1)
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)
        axes.grid(which="both")
        self._axes = axes
        self._ellipse_center = Point(*ELLIPSE_CENTER, name='CENTER')
        self._ellipse_top    = Point(*(self._ellipse_center + Point(0,int(ELLIPSE_HEIGHT/2))).get_point(), name='TOP')
        self._ellipse_right  = Point(*(self._ellipse_center + Point(int(ELLIPSE_WIDTH/2),0)).get_point(), name='RIGHT')

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
        #TODO: REFACTOR THIS!!!
        if self._ellipse:
            self._ellipse.remove()
        if self._center_annotation:
            self._center_annotation.remove()
        if self._top_annotation:
            self._top_annotation.remove()
        if self._right_annotation:
            self._right_annotation.remove()
        
        width  = 2 * ((self._ellipse_right - self._ellipse_center).get_point()[0])
        height = 2 * ((self._ellipse_top   - self._ellipse_center).get_point()[1])
        self._ellipse = Ellipse(xy=self._ellipse_center.get_point(), width=width, height=height,
                          edgecolor='r', fc='None', lw=2)
        self._axes.add_patch(self._ellipse)
        
        # draw markers
        self._center_annotation = self._axes.scatter(*self._ellipse_center.get_point(), marker='+', c='b', linewidths=5)
        self._top_annotation    = self._axes.scatter(*self._ellipse_top.get_point(),    marker='+', c='b', linewidths=5)
        self._right_annotation  = self._axes.scatter(*self._ellipse_right.get_point(),  marker='+', c='b', linewidths=5)
        
        self._figure.canvas.draw()

    def _find_neighbor_point(self, event):
        """ Find point around mouse position
        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 3.0
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for p in self._points:
            x, y = p.get_point()
            distance = math.hypot(event.xdata - x, event.ydata - y)
            if distance < min_distance:
                min_distance = distance
                nearest_point = p
        if min_distance < distance_threshold:
            return nearest_point
        return None

    def _on_click(self, event):
        """ callback method for mouse click event
        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point

        # TODO: do we need right-click?
        # right click
        # elif event.button == 3 and event.inaxes in [self._axes]:
        #     point = self._find_neighbor_point(event)
        #     if point:
        #         self._remove_point(*point)
        #         self._update_plot()

    def _on_release(self, event):
        """ callback method for mouse release event
        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        """ callback method for mouse motion event
        :type event: MouseEvent
        """
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._dragging_point.get_name() == 'CENTER':
            # move all points by same amount
            width, _ = (self._ellipse_center - self._ellipse_right).get_point()
            _ , height = (self._ellipse_top - self._ellipse_center).get_point()
            c_x, c_y = self._dragging_point.update_point(int(event.xdata), int(event.ydata))
            self._ellipse_right.update_point(c_x + abs(width), c_y)
            self._ellipse_top.update_point(c_x, c_y + height)
        
        elif self._dragging_point.get_name() == 'TOP':
            # move only top point, restrict to y axis
            x, _ = self._dragging_point.get_point()
            self._dragging_point.update_point(x, event.ydata)
        
        elif self._dragging_point.get_name() == 'RIGHT':
            # move only right point, restrict to x axis
            _, y = self._dragging_point.get_point()
            self._dragging_point.update_point(event.xdata, y)  
        
        self._update_plot()


if __name__ == "__main__":
    plot = DraggablePlotExample()