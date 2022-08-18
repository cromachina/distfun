import os
import pathlib

import glm
import imgui
import moderngl_window as mglw
import numpy as np
from moderngl_window.integrations.imgui import ModernglWindowRenderer

render_vert = '''
#version 460
in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
'''

class FileWatch():
    def __init__(self, path, on_changed):
        self.stamp = 0
        self.path = pathlib.Path(path)
        self.on_changed = on_changed

    def check(self):
        stamp = os.stat(self.path).st_mtime
        if stamp != self.stamp:
            self.stamp = stamp
            self.on_changed(self.path)

class Distfun(mglw.WindowConfig):
    gl_version = 4,6
    title = "Distfun"
    window_size = 800, 800
    aspect_ratio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.wnd.exit_key = False

        # Input data
        self.keys = set()

        # Scene data
        self.position = glm.vec3(-0.166, 2.6, -1.945)
        self.pitch = -0.435
        self.yaw = 3.487

        self.epsilon = 0.001
        self.epsilon_delta = 0.0001
        self.fov = 90
        self.frame = 0
        self.move_speed = 0.05
        self.turn_speed = 0.015

        # Render data
        self.render_program = None
        self.file_watch = FileWatch('scene.frag', self.reload_shaders)
        self.file_watch.check()
        self.render_buffer = self.ctx.buffer(np.array([[-1, -1], [3, -1], [-1, 3]], dtype=np.float32))
        self.render_vao = self.ctx.vertex_array(self.render_program, self.render_buffer, 'position')

    def reload_shaders(self, path):
        try:
            print('loading shader:', path)
            new_prog = self.ctx.program(vertex_shader=render_vert, fragment_shader=path.read_text())
            print(self.ctx.error)
            if self.render_program:
                self.render_program.release()
            self.render_program = new_prog
        except Exception as e:
            print(e)

    def process_input(self, time, frametime):
        if not self.imgui.io.want_capture_keyboard:
            if self.wnd.keys.ESCAPE in self.keys:
                self.wnd.mouse_exclusivity = False
            if self.wnd.keys.Q in self.keys:
                self.position.y -= self.move_speed
            if self.wnd.keys.E in self.keys:
                self.position.y += self.move_speed
            move_vec = glm.vec3(0)
            if self.wnd.keys.W in self.keys:
                move_vec += glm.vec3(0, 0, 1)
            if self.wnd.keys.S in self.keys:
                move_vec += glm.vec3(0, 0, -1)
            if self.wnd.keys.A in self.keys:
                move_vec += glm.vec3(-1, 0, 0)
            if self.wnd.keys.D in self.keys:
                move_vec += glm.vec3(1, 0, 0)
            if move_vec != glm.vec3(0):
                self.position += glm.normalize(glm.rotateY(move_vec, self.yaw + glm.pi())) * self.move_speed

    def render_ui(self, time, frametime):
        imgui.new_frame()
        imgui.set_next_window_position(0, 0)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0.5)
        imgui.begin('Info', False, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS |
            imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_ALWAYS_AUTO_RESIZE |
            imgui.WINDOW_NO_TITLE_BAR)
        imgui.text(f'Frame: {self.frame}')
        imgui.text(f'Time: {time:.3f}')
        imgui.text(f'Frametime: {frametime:.3f}')
        imgui.text(f'Pos: {self.position.x:.3f}, {self.position.y:.3f}, {self.position.z:.3f}')
        imgui.text(f'View(pitch,yaw): {self.pitch:.3f}, {self.yaw:.3f}')
        imgui.end()
        imgui.pop_style_color()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def render(self, time, frametime):
        self.file_watch.check()
        self.process_input(time, frametime)
        if self.render_program:
            view_matrix = glm.translate(self.position)
            view_matrix = view_matrix * glm.mat4(glm.quat(glm.vec3(self.pitch, self.yaw, 0.0)))
            self.render_program['view_matrix'].write(view_matrix)
            self.render_program['resolution'] = self.window_size
            self.render_program['epsilon'] = self.epsilon
            self.render_program['fov'] = self.fov
            self.render_program['max_steps'] = 100
            #self.render_program['time'] = time
            #self.render_program['frame'] = self.frame
            self.render_vao.render()
        self.render_ui(time, frametime)
        self.frame += 1

    def resize(self, width, height):
        self.window_size = self.wnd.size
        self.imgui.resize(width, height)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            self.keys.add(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self.keys.remove(key)

        if not self.wnd.mouse_exclusivity:
            self.imgui.key_event(key, action, modifiers)

    def mouse_position_event(self, x, y, dx, dy):
        if not self.wnd.mouse_exclusivity:
            self.imgui.mouse_position_event(x, y, dx, dy)
        else:
            self.pitch = glm.clamp(self.pitch + self.turn_speed * -dy, -glm.half_pi(), glm.half_pi())
            self.yaw = self.yaw + self.turn_speed * dx

    def mouse_drag_event(self, x, y, dx, dy):
        if not self.wnd.mouse_exclusivity:
            self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        if not self.wnd.mouse_exclusivity:
            self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        if not self.imgui.io.want_capture_mouse:
            self.wnd.mouse_exclusivity = True
        if not self.wnd.mouse_exclusivity:
            self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        if not self.wnd.mouse_exclusivity:
            self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        if not self.wnd.mouse_exclusivity:
            self.imgui.unicode_char_entered(char)

if __name__ == '__main__':
    mglw.run_window_config(Distfun)
