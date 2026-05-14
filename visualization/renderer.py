import pygame
import numpy as np

class Renderer:
    def __init__(self, grid_size=10, cell_size=50):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_size = grid_size * cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Tactical AI Game Strategy")
        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 200, 50)
        self.GOLD = (255, 215, 0)

    def render(self, env):
        self.screen.fill(self.WHITE)

        # Draw grid
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size, y))

        # Draw objective
        obj_pos = env.objective_pos
        pygame.draw.circle(
            self.screen, self.GOLD, 
            (obj_pos[0] * self.cell_size + self.cell_size // 2, obj_pos[1] * self.cell_size + self.cell_size // 2), 
            self.cell_size // 3
        )

        # Draw agents
        for agent in env.agents:
            pos = env.agent_positions[agent]
            health = env.agent_health[agent]
            color = self.BLUE if "blue" in agent else self.RED
            
            rect = pygame.Rect(
                pos[0] * self.cell_size + 5, 
                pos[1] * self.cell_size + 5, 
                self.cell_size - 10, 
                self.cell_size - 10
            )
            pygame.draw.rect(self.screen, color, rect)
            
            # Health bar
            health_width = (self.cell_size - 10) * (health / 100)
            pygame.draw.rect(
                self.screen, self.GREEN, 
                (pos[0] * self.cell_size + 5, pos[1] * self.cell_size + 2, health_width, 4)
            )

        pygame.display.flip()
        self.clock.tick(10) # 10 FPS

    def close(self):
        pygame.quit()
