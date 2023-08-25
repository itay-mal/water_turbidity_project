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

    def __add__(self, other, _name=None):
        x = self._x + other._x
        y = self._y + other._y
        return Point(x, y, name=_name)
    
    def __sub__(self, other, _name=None):
        x = self._x - other._x
        y = self._y - other._y
        return Point(x, y, name=_name)
    
    def __repr__(self):
        return "Point {} {}".format(self.get_point(), self._name)
    
    def get_name(self):
        return self._name

class MyEllipse():
    def __init__(self, x, y, w, h, color='b', name="MyEllipse"):
        """
        x,y - ellipse center coordinates [pixels]
        w - width [pixels]
        h - height [pixels]
        """
        self._center = Point(x,y, name='CENTER')
        self._top    = Point(*(self._center + Point(0, int(h/2))).get_point(), name='TOP') 
        self._right  = Point(*(self._center + Point(int(w/2), 0)).get_point(), name='RIGHT')
        self._points = (self._center,
                        self._right,
                        self._top)

        self._patch  = None 
        self._anchors_scatter = None
        self._color = color
        self._name = name
    
    def __repr__(self):
        return "{}".format(self._name)

    def get_points(self):
        return self._points    

    def update_annotations(self, axes):
        """
        update all ellipse drawn elements
        should be called after points are updated
        axes - the subplot to be updated
        """
        if self._patch:
            self._patch.remove()
        if self._anchors_scatter:
            self._anchors_scatter.remove()
        
        width  = 2 * ((self._right - self._center).get_point()[0])
        height = 2 * ((self._top   - self._center).get_point()[1])
        self._patch = Ellipse(xy=self._center.get_point(), width=width, height=height,
                              edgecolor=self._color, fc='None', lw=2)
        axes.add_patch(self._patch)
        x_c, y_c = self._center.get_point()
        x_r, y_r = self._right.get_point()
        x_t, y_t = self._top.get_point()

        # draw markers
        self._anchors_scatter = axes.scatter((x_c,x_r,x_t),(y_c,y_r,y_t), marker='+', c='b', linewidths=5)
        
    def update_points(self, point, event):
        """
        updates the other ellipse points location according to dragged point
        point - the dragged point
        event - the motion event (mouse poisition)
        """
        if point.get_name() == 'CENTER':
            # move all points by same amount
            width, _ = (self._center - self._right).get_point()
            _ , height = (self._top - self._center).get_point()
            c_x, c_y = point.update_point(int(event.xdata), int(event.ydata))
            self._right.update_point(c_x + abs(width), c_y)
            self._top.update_point(c_x, c_y + height)
        
        elif point.get_name() == 'TOP':
            # move only top point, restrict to y axis
            x, _ = point.get_point()
            point.update_point(x, event.ydata)
        
        elif point.get_name() == 'RIGHT':
            # move only right point, restrict to x axis
            _, y = point.get_point()
            point.update_point(event.xdata, y)
        
class DraggablePlotExample(object):
    """ An example of plot with draggable markers """

    def __init__(self):
        self._figure, self._axes = None, None
        self._ellipse = None
        
        self._dragging_point_target = None

        self._targets = (MyEllipse(30, 50, 60, 40, 'r', "target1"),
                         MyEllipse(70, 50, 60, 40, 'b', "target2"))
        
        
        self._init_plot()

    def _init_plot(self):
        self._figure = plt.figure("Example plot")
        axes = plt.subplot(1, 1, 1)
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)
        axes.grid(which="both")
        self._axes = axes
        
        self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._update_plot()
        plt.show()

    def _update_plot(self):
        for t in self._targets:
            t.update_annotations(self._axes)        
        self._figure.canvas.draw()

    def _find_neighbor_point(self, event):
        """ Find point around mouse position
        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 3.0
        nearest_point, nearest_target = None, None
        min_distance = math.sqrt(2 * (100 ** 2))
        for t in self._targets:
            for p in t.get_points():
                x, y = p.get_point()
                distance = math.hypot(event.xdata - x, event.ydata - y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point, nearest_target = p, t

        if min_distance < distance_threshold:
            return nearest_point, nearest_target
        return None

    def _on_click(self, event):
        """ callback method for mouse click event
        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point, target = self._find_neighbor_point(event)
            if point:
                self._dragging_point_target = point, target

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
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point_target:
            self._dragging_point_target = None
            self._update_plot()

    def _on_motion(self, event):
        """ callback method for mouse motion event
        :type event: MouseEvent
        """
        if not self._dragging_point_target:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        drag_point, drag_ellipse = self._dragging_point_target
        drag_ellipse.update_points(drag_point, event)

        self._update_plot()


if __name__ == "__main__":
    plot = DraggablePlotExample()