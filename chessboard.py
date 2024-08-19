from typing import Dict

from PIL import Image


class Chessboard:

    def __init__(self, width: int = 8, height: int = 8, step_cost: int = -1) -> None:
        """

        Args:
            width (int, optional): Width of chessboard. Defaults to 8.
            height (int, optional): Height of chessboard. Defaults to 8.
            step_cost (int, optional): Reward of step. Defaults to -1.
        """
        self.width = width
        self.height = height
        self.step_cost = step_cost
        self.states_rewards = self._create_rectangle_chessboard()

    def _create_rectangle_chessboard(self) -> Dict[tuple[int, int], int]:
        """Create rectangular chessboard with defined width and height.
        Chessboard is represented as a dictionary of tuples which consist
        of coordinates (x, y) and value which represent cost of step.

        Returns:
            Dict[tuple[int, int], int]: Created chessboard with cost of step.
        """
        chessboard = {}
        for i in range(0, self.width):
            for j in range(0, self.height):
                chessboard[(i, j)] = self.step_cost
        return chessboard

    def modify_chessboard(self) -> None:
        """Manual modification of step cost for selected states. User can modify this method by his will.
        Be aware to modify states which are included in chessboard!
        """
        #  deadly states
        self.states_rewards[(5, 5)] = -100
        self.states_rewards[(5, 6)] = -100
        self.states_rewards[(6, 5)] = -100
        self.states_rewards[(6, 6)] = -100
        self.states_rewards[(8, 4)] = -100
        self.states_rewards[(8, 5)] = -100
        self.states_rewards[(9, 4)] = -100
        self.states_rewards[(9, 5)] = -100
        self.states_rewards[(8, 1)] = -100
        self.states_rewards[(8, 2)] = -100
        self.states_rewards[(8, 8)] = -100
        self.states_rewards[(8, 9)] = -100
        self.states_rewards[(8, 10)] = -100
        self.states_rewards[(8, 11)] = -100
        self.states_rewards[(8, 12)] = -100
        self.states_rewards[(8, 13)] = -100
        self.states_rewards[(8, 14)] = -100
        self.states_rewards[(9, 6)] = -100
        self.states_rewards[(3, 2)] = -100
        self.states_rewards[(3, 3)] = -100
        self.states_rewards[(3, 5)] = -100
        self.states_rewards[(7, 2)] = -100
        self.states_rewards[(13, 12)] = -100

        for i in range(0, self.width):
            self.states_rewards[(i, 7)] = -100
        for i in range(0, self.height):
            self.states_rewards[(4, i)] = -100

    def draw_users_chessboard(self) -> None:
        """Draw chessboard modified by user and save it as "Chessboard.jpg".
        """
        img_of_standard_chessboard = self.draw_standard_chessboard()
        self.img_of_modified_chessboard = self._draw_modified_chessboard(img_of_standard_chessboard)
        self.img_of_modified_chessboard.resize(
                                        size=(1_000, 1_000),
                                        resample=Image.NEAREST).transpose(Image.FLIP_TOP_BOTTOM).save("Chessboard.jpg")

    def draw_standard_chessboard(self) -> Image:
        """Draw standard chessboard.

        Returns:
            Image: Image of standard chessboard.
        """
        img_of_standard_chessboard = Image.new(mode="RGB", size=(self.width, self.height))
        pixels = img_of_standard_chessboard.load()
        color = [(0, 0, 0), (255, 255, 255)]
        actual_default_color = True
        first_color = actual_default_color
        for i in range(0, self.width):
            first_color = actual_default_color
            for j in range(0, self.height):
                pixels[i, j] = color[actual_default_color]
                actual_default_color = not actual_default_color
            actual_default_color = not first_color
        return img_of_standard_chessboard

    def _draw_modified_chessboard(self, img_of_standard_chessboard: Image) -> Image:
        """Redraw standard chessboard according to users setting of rewards.

        Args:
            img_of_standard_chessboard (Image): Image of standard chessboard.

        Returns:
            Image: Image of modified chessboard.
        """
        pixels = img_of_standard_chessboard.load()
        for state, reward in list(self.states_rewards.items()):
            if reward == -100:
                pixels[state[0], state[1]] = (255, 0, 0)
        return img_of_standard_chessboard
