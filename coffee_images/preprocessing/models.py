import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

class MultiPolygonMaskBuilder:
    def __init__(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.dtype == object:
            raise ValueError("Image array must not have dtype=object")
        self.image = image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.polygons = []
        self.current_selector = PolygonSelector(self.ax, self.onselect, props=dict(color='green'), handle_props=dict(color='green'))
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("Instructions:")
        print("- Draw a polygon by clicking points. Double-click to finish each polygon.")
        print("- Press 'n' to start a new polygon.")
        print("- Press 'm' to show mask.")
        print("- Press 'q' to quit.")
        print("- Press 'e' to exit.")
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.exit_requested = False
        plt.show()

    def onselect(self, verts):
        self.polygons.append(verts)
        # Draw the current polygon on the image
        xs, ys = zip(*verts)
        self.ax.plot(xs + (xs[0],), ys + (ys[0],), color='red')
        self.fig.canvas.draw_idle()
        print(f"Polygon {len(self.polygons)} added. Press 'n' for new, 'm' to show mask, 'q' to quit.")

    def on_key_press(self, event):
        if event.key == 'n':
            # Start a new polygon
            self.current_selector.disconnect_events()
            self.current_selector = PolygonSelector(self.ax, self.onselect, props=dict(color='green'), handle_props=dict(color='green'))
        elif event.key == 'm':
            # Show mask
            self.update_mask()
            plt.figure()
            plt.imshow(self.mask, cmap='gray')
            plt.title('Combined Mask')
            plt.show()
        elif event.key == 'q':
            plt.close(self.fig)
            self.fig.canvas.mpl_disconnect(self.cid)
        elif event.key == 'e':
            plt.close(self.fig)
            self.fig.canvas.mpl_disconnect(self.cid)
            self.exit_requested = True

    def update_mask(self):
        # Fill all polygons in mask
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for verts in self.polygons:
            path = Path(verts)
            ny, nx = self.image.shape[:2]
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            grid = path.contains_points(points)
            poly_mask = grid.reshape((ny, nx)).astype(np.uint8) * 255
            mask = np.maximum(mask, poly_mask)
        self.mask = mask
