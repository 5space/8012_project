import numpy as np

import pygame
import pygame_gui

G = 0.8

COLORS = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0)]

TOPRIGHT_ANCHORS = {"left": "right",
                    "right": "right",
                    "top": "top",
                    "bottom": "top"}

BOTTOMRIGHT_ANCHORS = {"left": "right",
                       "right": "right",
                       "top": "bottom",
                       "bottom": "bottom"}

np.set_printoptions(precision=20)

class Simulation:

    def __init__(self, masses=[], bodies=[]):
        self.masses = masses
        self.bodies = bodies

        self.t = 0
        self.n = len(self.masses)
        self.running = True
    
    def add_body(self, mass, pos, vel):
        body = np.array([pos, vel])
        self.masses.append(mass)
        self.bodies.append(body)
        self.n += 1
    
    def set_body(self, index, mass, pos, vel):
        body = np.array([pos, vel])
        self.masses[index] = mass
        self.bodies[index] = body
    
    def remove_body(self, index):
        self.masses.pop(index)
        self.bodies.pop(index)
        self.n -= 1
    
    def dv_dt(self, i, body=None):
        if body is None: body = self.bodies[i]

        F = np.zeros(3)
        for j in range(self.n):
            if i == j: continue
            offset = self.bodies[j][0] - self.bodies[i][0]
            F += G * self.masses[j] * offset / np.linalg.norm(offset) ** 3
        return F

    def ode(self, i, body=None):
        if body is None: body = self.bodies[i]
        return np.array([body[1], self.dv_dt(i, body)])
    
    def step_euler(self, dt):
        new = [None for _ in range(self.n)]
        for i in range(self.n):
            new[i] = self.bodies[i] + self.ode(i) * dt
        self.bodies = new
        self.t += dt
    
    def step_sieuler(self, dt):
        new = [None for _ in range(self.n)]
        for i in range(self.n):
            y0 = self.bodies[i] + self.ode(i) * dt
            r1, v1 = y0
            r1 += self.ode(i)[1] * dt * dt
            new[i] = np.array([r1, v1]).astype(np.single).astype(np.double)
        self.bodies = new
        self.t += dt

    # def step_modifiedeuler(self, dt):
    #     new = [None for _ in range(self.n)]
    #     for i in range(self.n):
    #         y0 = self.bodies[i]
    #         k1 = self.ode(i, y0)
    #         k2 = self.ode(i, y0 + dt*k1)
    #         new[i] = y0 + dt/2 * (k1 + k2)
    #     self.bodies = new
    #     self.t += dt
    
    # def step_rungekutta(self, dt):
    #     new = [None for _ in range(self.n)]
    #     for i in range(self.n):
    #         y0 = self.bodies[i]
    #         k1 = self.ode(i, y0)
    #         k2 = self.ode(i, y0 + dt*k1/2)
    #         k3 = self.ode(i, y0 + dt*k2/2)
    #         k4 = self.ode(i, y0 + dt*k3)
    #         new[i] = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    #     self.bodies = new
    #     self.t += dt
    
    def energy(self):
        e = 0
        amt = self.n
        for i in range(amt):
            e += self.masses[i] * np.linalg.norm(self.bodies[i][1]) ** 2 / 2
            for j in range(i+1, amt):
                e -= G * self.masses[i] * self.masses[j] / np.linalg.norm(self.bodies[j][0] - self.bodies[i][0])
        return e
    
    def center_of_mass(self):
        return sum(self.masses[i] * self.bodies[i][0] for i in range(self.n)) / sum(self.masses)
    
    def linear_momentum(self):
        return sum(self.masses[i] * self.bodies[i][1] for i in range(self.n))
    
    def angular_momentum(self, reference_frame=np.zeros([2, 3])):
        return sum(self.masses[i] * np.cross(*(self.bodies[i] - reference_frame)) for i in range(self.n))[2]
    
    def to_xyz(self):
        return tuple([b[0][i] for b in self.bodies] for i in range(3))
        
sim = Simulation()
sim.add_body(1, [0, 0, 0], [0, 0, 0])
sim.add_body(1, [1, 0, 0], [0, 1, 0])
sim.add_body(1, [-1, 0, 0], [0, -1, 0])

"""
Euler: v^2 = G/r * 5/4
Lagrange: v^2 = G/r * 1/sqrt(3)
"""

pygame.init()
WIDTH = 640
HEIGHT = 480
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True

ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT), "theme.json")

algorithm_label = pygame_gui.elements.UILabel(pygame.Rect(5, 5, 80, 25), "Algorithm:", ui_manager)
algorithm = pygame_gui.elements.UIDropDownMenu(["Euler", "Mod. Euler", "SI Euler", "Runge-Kutta"], "SI Euler", pygame.Rect(90, 5, 120, 25), ui_manager)

solution_label = pygame_gui.elements.UILabel(pygame.Rect(5, 35, 80, 25), "Solution:", ui_manager)
solution = pygame_gui.elements.UIDropDownMenu(["", "Euler", "Lagrange", "Figure-8"], "", pygame.Rect(90, 35, 120, 25), ui_manager)

grav_label = pygame_gui.elements.UILabel(pygame.Rect(-280, 5, 70, 25), "G = {0:.2f}".format(G), ui_manager, anchors=TOPRIGHT_ANCHORS)
grav_slider = pygame_gui.elements.UIHorizontalSlider(pygame.Rect(-205, 5, 200, 25), 100*G, (0, 200), ui_manager, anchors=TOPRIGHT_ANCHORS)

playpause_button = pygame_gui.elements.UIButton(pygame.Rect(-65, -30, 60, 25), "Pause", ui_manager, anchors=BOTTOMRIGHT_ANCHORS)
ui_visible = True

def toggle_running():
    sim.running = not sim.running
    if sim.running:
        playpause_button.set_text("Pause")
    else:
        playpause_button.set_text("Play")

def get_hovered_body():
    return 0


while running:
    dt = clock.tick(120)/1000.0
    if dt >= 0.03:
        dt = 0.03

    if sim.running:
        sim.step_sieuler(dt)
    
    for event in pygame.event.get():
        ui_manager.process_events(event)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.size
            ui_manager.set_window_resolution((WIDTH, HEIGHT))
        elif event.type == pygame.USEREVENT:
            if event.ui_element == grav_slider:
                G = event.value / 100
                grav_label.set_text("G = {0:.2f}".format(G))
            elif event.ui_element == solution and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                print(event)
            elif event.ui_element == playpause_button and event.user_type == pygame_gui.UI_BUTTON_START_PRESS:
                toggle_running()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                ui_visible = not ui_visible
            elif event.key == pygame.K_SPACE:
                toggle_running()
    
    if ui_visible:
        ui_manager.update(dt)

    screen.fill((0, 0, 0))
    for i, (pos, _) in enumerate(sim.bodies):
        x, y, z = pos * HEIGHT/3 + np.array([WIDTH/2, HEIGHT/2, 0])
        radius = (16*sim.masses[i] + 80)/(sim.masses[i] + 20)
        pygame.draw.circle(screen, COLORS[i], (x, y), radius)
    
    if ui_visible:
        ui_manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()