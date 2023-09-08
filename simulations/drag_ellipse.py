import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from matplotlib.backend_bases import MouseEvent
from skimage import io


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
    def __init__(self, x, y, w, h, theta=0, color='b', name="MyEllipse"):
        """
        x,y - ellipse center coordinates [pixels]
        w - width [pixels]
        h - height [pixels]
        """
        self._center = Point(x,y, name='CENTER')
        self._top    = Point(*(self._center - Point(0, h)).get_point(), name='TOP')  # subtract because in images y axis is upside down
        self._right  = Point(*(self._center + Point(w, 0)).get_point(), name='RIGHT')
        self._points = (self._center,
                        self._right,
                        self._top)

        self._theta = theta

        self._patch  = None 
        self._anchors_scatter = None
        self._color = color
        self._name = name
    
    def get_params(self):
        x_center, y_center = self._center.get_point()
        _, y_top = self._top.get_point()
        x_right, _ = self._right.get_point()
        width = 2 * abs(x_right - x_center)
        height = 2 * abs(y_top - y_center)
        return x_center, y_center, width, height
    
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
        updates the ellipse points location according to dragged point
        point - the dragged point
        event - the motion event (mouse poisition)
        """
        if point.get_name() == 'CENTER':
            # move all points by same amount
            width, _ = (self._center - self._right).get_point()
            _ , height = (self._top - self._center).get_point()
            c_x, c_y = point.update_point(event.xdata, event.ydata)
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
        
class DraggablePlot(object):
    """plot with draggable markers"""

    def __init__(self, 
                 image=None, 
                 target1_est=(30,50,60,40,0),
                 target2_est=(70,50,60,40,0),
                 my_callback=None,
                 standalone = True):
        """
        image - will be displayed on background
        target1/2_est - estimated ellipse params in format (x_enter, y_center, width, height) [pixels]
        my_callback - function pointer, will be called with args: (((x1,y1),w1,h1,0),((x2,y2),w2,h2,0),img) when 'c' pressed 
                      0 is currently hard coded as angle # TODO: do we want to keep it that way?
        standalone - running in standalone mode (false -> running from TkInter)
        """
        self._figure, self._axes = None, None
        self._ellipse = None
        self.standalone = standalone
        self._dragging_point_target = None

        self._targets = (MyEllipse(*target1_est, 'r', "target1"),
                         MyEllipse(*target2_est, 'b', "target2"))
        
        self._my_callback = my_callback
        self.image = image
        self.coeffs = None

    def get_axes(self):
        try:
            return self._axes
        except:
            print("axes not yet exist, run _init_plot first")
        
    
    def get_coeffs(self):
        if self.coeffs:
            return self.coeffs
        else:
            print("coeffs not yet exist, run calculation first")
        
    def _init_plot(self, figure=None, canvas=None):
        self._figure = plt.figure("Example plot") if figure is None else figure
        axes = plt.subplot(1, 1, 1) if figure is None else figure.add_subplot()
        if self.image is None:
            axes.set_xlim(0, 100)
            axes.set_ylim(0, 100)
            axes.grid(which="both")
        self._axes = axes
        if self.image is not None:
            self._axes.imshow(self.image)
        self.myCanvas = self._figure.canvas if canvas is None else canvas
        self.myCanvas.mpl_connect('button_press_event', self._on_click)
        self.myCanvas.mpl_connect('button_release_event', self._on_release)
        self.myCanvas.mpl_connect('motion_notify_event', self._on_motion)
        self.myCanvas.mpl_connect('key_press_event', self._on_key_press)
        self._update_plot()
        if self.standalone:
            self._axes.set_title("correct targets and press \'c\'")
            plt.show()
        else:
            self._axes.set_title("correct targets and press \'Calculate\'")

    def _update_plot(self):
        for t in self._targets:
            t.update_annotations(self._axes)        
        self.myCanvas.draw()

    def _find_neighbor_point(self, event):
        """ 
        Find point around mouse position
        :rtype: ((int, int)|None)
        :return: (x, y), target if there are any point around mouse else None, None
        """
        distance_threshold = 10
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
        return None, None

    def _on_click(self, event):
        """ 
        callback method for mouse click event
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
        """ 
        callback method for mouse release event
        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point_target:
            self._dragging_point_target = None
            self._update_plot()

    def _on_motion(self, event):
        """ 
        callback method for mouse motion event
        :type event: MouseEvent
        """
        if not self._dragging_point_target:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        drag_point, drag_ellipse = self._dragging_point_target
        drag_ellipse.update_points(drag_point, event)

        self._update_plot()

    def _on_key_press(self, event):
        if event.key == 'c':
            if self._my_callback:
                x1, y1, w1, h1 = self._targets[0].get_params()
                x2, y2, w2, h2 = self._targets[1].get_params()
                self.coeffs = self._my_callback(((x1, y1), w1, h1, 0), ((x2, y2), w2, h2, 0), self.image, show_mask=self.standalone)
            else:
                print("\'c\' is pressed but no callback defined")

if __name__ == "__main__":
    # TODO: remove this and implement as part of the complete algorithm
    img = io.imread('C:/Users/itaym/Desktop/000.png')  
    
    plot = DraggablePlot(my_callback=print, image=img)._init_plot()