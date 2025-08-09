import pygame
import numpy as np
from scipy.signal import convolve2d
from settings import *
from datetime import datetime
import logging
from tkinter import filedialog


logger = logging.getLogger(__name__)
logging.basicConfig(filename='../logs/basic_config.log',
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')


class App:
    def __init__(self, cell_size:int=10,
                 number_of_cells_horizontal:int=100,
                 number_of_cells_vertical:int=60,
                 living_cell_symbol:str="█") -> None:
        """
        Constructor of the class App.
        """
        self.cell_size = cell_size
        self.num_cells_x = number_of_cells_horizontal
        self.num_cells_y = number_of_cells_vertical
        self.grid_width = self.cell_size * self.num_cells_x
        self.grid_height = self.cell_size * self.num_cells_y
        if len(living_cell_symbol) != 1:
            logging.info("Living cell symbol of size different than 1. Changing it to a safe symbol.")
            self.full_cell = "█"
        else:
            self.full_cell = living_cell_symbol if living_cell_symbol else "█" # \u2588 or U+2588 by default
        self.empty_cell = " " # empty spaces will be saved as just space symbols

        # setting up pygame and useful variables
        pygame.init()

        try:
            logger.info("Attempt at loading an icon image.")
            icon = pygame.image.load('../graphics/conway.jpg')
            pygame.display.set_icon(icon)
        except pygame.error as e:
            logger.info("Encountered an error loading the icon.")
            logger.error(e)

        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()

        try:
            logger.info("Attempt at loading a music file.")
            pygame.mixer.music.load('../music/sans..mp3')
            pygame.mixer.music.play(-1)
        except pygame.error as e:
            logger.info("Encountered an error loading the music file.")
            logger.error(e)

        pygame.display.set_caption("Conway's Game of Life")
        self.screen = pygame.display.set_mode((self.grid_width, self.grid_height))

        self.cells = np.zeros((self.num_cells_y, self.num_cells_x))
        self.generation = 0
        self.population = 0
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.clock = pygame.time.Clock()
        self.running = False

    def update(self) -> tuple[np.ndarray, int]:
        """
        Updates the state of the cells using convolution and draws them on the screen.
        """

        # calculate neighbor counts for all cells using convolution
        kernel = np.ones((3, 3), dtype=int)
        kernel[1, 1] = 0
        neighbor_counts = convolve2d(self.cells, kernel, mode='same', boundary='wrap')

        # apply the rules to get the next generation
        updated_cells = np.zeros_like(self.cells)
        updated_cells[(self.cells == 1) & ((neighbor_counts == 2) | (neighbor_counts == 3))] = 1
        updated_cells[(self.cells == 0) & (neighbor_counts == 3)] = 1

        # draw the cells
        for row, col in np.ndindex(self.cells.shape):
            color = COLOR_ALIVE_NEXT if self.cells[row, col] == 1 else COLOR_BG
            pygame.draw.rect(self.screen, color, (col * self.cell_size, row * self.cell_size, self.cell_size - 1, self.cell_size - 1))

        # the population is the count of currently living cells
        population = int(np.sum(self.cells))

        return updated_cells, population


    def save_grid(self, filename, encoding="utf-8"):
        """
        Method reads 0s and 1s from the program and safes them to a file as self.full_cell and self.empty_cell symbols respectively.
        """
        with open(filename, 'w') as f:
            if self.full_cell != '#':
                try:
                    logging.info("Attempt at saving into a file using a 'U+2588' as full block.")
                    for row in self.cells:
                        f.write(''.join(self.full_cell if cell == 1 else self.empty_cell for cell in row) + '\n')
                except Exception as ex:
                    logging.info("Could not save to a file using 'U+2588' as full block.")
                    logging.error(ex)
                    logging.info("Changing full block symbol to '#'.")
                    self.full_cell = "#"
            if self.full_cell == '#':
                logging.info("Saving into a file using a '#' as full block.")
                for row in self.cells:
                    f.write(''.join(self.full_cell if cell == 1 else self.empty_cell for cell in row) + '\n')


    def load_grid(self, filename, encoding="utf-8"):
        """
        Method reads each self.full_cell and self.empty_cell symbols as used in save (which can cause errors when loading files
        encoded with different keys) as respectively 1 and 0.
        """
        with open(filename, 'r') as f:
            self.cells = np.array([[0 if char == self.empty_cell else 1 for char in line.rstrip('\n')] for line in f])
            self.generation = 0

    def run(self) -> None:
        """
        Main loop of the program.
        """

        while True:
            # event control
            for event in pygame.event.get():
                # escaping the game
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                # pausing and resuming the game
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.running = not self.running
                    # screenshot of the grid
                    elif event.key == pygame.K_s and not self.running:
                        self.save_grid(f"../saved_grids/grid_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}_gen{self.generation}_pop{self.population}.txt")
                    # load an existing file
                    elif event.key == pygame.K_l and not self.running:
                        try:
                            logging.info(f"Attempt at accesing files within 'saved_grids' folder.")
                            file_path = filedialog.askopenfilename(
                                title="Select a file",
                                initialdir=f"../saved_grids",
                                filetypes=(("Text files", "*.txt"),)
                            )
                            self.load_grid(file_path)
                        except Exception as ex:
                            logging.info(f"Could not open a file from 'saved_grids' folder.")
                            logging.error(ex)
                # adding cells to the grid
                elif pygame.mouse.get_pressed()[0] and not self.running:
                    pos = pygame.mouse.get_pos()
                    self.cells[pos[1] // self.cell_size, pos[0] // self.cell_size] = 1

            # display background
            self.screen.fill(COLOR_GRID)
            cells_to_draw, population_to_display = self.update()

            # display grid and current configuration
            if self.running:
                self.cells = cells_to_draw
                self.population = population_to_display
                self.generation += 1
            else:
                # recalculate population on pause for accuracy
                self.population = int(np.sum(self.cells))

            # displaying counters
            generation_text = self.font.render(f'Generation {self.generation}', True, COLOR_TEXT)
            population_text = self.font.render(f'Population {self.population}', True, COLOR_TEXT)
            self.screen.blit(generation_text, (5, 5))
            self.screen.blit(population_text, (5, 35))

            # wait a bit so that the user can observe and analyze the changes
            if self.running:
                self.clock.tick(10)

            pygame.display.flip()
