import gizeh
import moviepy.editor as mpy

W, H = 256, 256  # width, height, in pixels
duration = 10  # duration of the clip, in seconds
radius = 10


def make_frame(t):
    surface = gizeh.Surface(W, H, bg_color=(.8, .8, .8))
    x = t*W/duration + radius
    circle = gizeh.circle(r=radius, xy=(x, H / 2), fill=(1, 0, 0))
    rect = gizeh.rectangle(lx=1, ly=H, xy=(int(0.7*W), H/2), fill=(0, 0, 0))
    circle.draw(surface)
    rect.draw(surface)
    return surface.get_npimage()


clip = mpy.VideoClip(make_frame, duration=duration)
clip.write_videofile("circle.mp4", fps=24)
