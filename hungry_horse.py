from typing import Dict, List

import pandas as pd
import numpy as np


class HungryHorse:

    def __init__(self, chessboard, starting_state: tuple[int, int] = (0, 0),
                 alpha: float = 0.01, eps: float = 0.05) -> None:
        self.chessboard = chessboard
        self.starting_state = starting_state
        self.visited_states = []
        self.possible_actions = self._generate_possible_actions()
        self.states_actions_values = self._create_combinations_of_all_possible_states_and_actions()

        self.alpha = alpha
        self.eps = eps
        self.gamma = 0.9
        self.reward = 1

    def _generate_possible_actions(self) -> Dict[str, np.ndarray[int, int]]:
        """Create dictionary of possible actions.
        Keys is a shortcut for a name of an action such as:
            ul = up left
            ur = up right
            ru = right up
            rd = right down
            dr = down right
            dl = down left
            ld = left down
            lu = left up
        Value is a vector which can be added to the actual position to move in to a new position.

        Returns:
            Dict[str, np.ndarray[int, int]]: Dictionary of possible actions.
        """
        return {"ul": np.array([-1, 2]),
                "ur": np.array([1, 2]),
                "ru": np.array([2, 1]),
                "rd": np.array([2, -1]),
                "dr": np.array([1, -2]),
                "dl": np.array([-1, -2]),
                "ld": np.array([-2, -1]),
                "lu": np.array([-2, 1])}

    def _create_combinations_of_all_possible_states_and_actions(self) -> Dict[tuple[int, int], Dict[str, float]]:
        """Create combinations of all possible states and actions. Results are stored as a Dict of dictionaries.
        The first key is meant to be coordinates of actual state. The value is inner dictionary where key is name of
        action and value is its Q(state, action) value initialized to be zero.

        Returns:
            Dict[tuple[int, int], Dict[str, float]]: All Q(state, action) values initialized to be zero.
        """
        states_actions_values = {}
        for state in self.chessboard:
            valid_actions = {}
            for action_name, action_vector in list(self.possible_actions.items()):
                if self._action_valid(state, action_vector):
                    valid_actions[action_name] = 0
            states_actions_values[state] = valid_actions
        return states_actions_values

    def _action_valid(self, state: tuple[int, int], action_vector: np.ndarray[int, int]) -> bool:
        """Check if action can be done in provided state.
        If final state is included in chessboard and not included in visited_states return True else False.

        Args:
            state (tuple[int, int]): Starting state on a chessboard.
            action_vector (np.ndarray[int, int]): Vector of movement for selected action.

        Returns:
            bool: If action is valid return True else False.
        """
        final_state = tuple((np.array(state) + action_vector).tolist())
        if final_state in self.chessboard and final_state not in self.visited_states:
            return True
        return False

    def run_sarsa(self):
        episode_score = {}
        for episode_number in range(0, 10_000):
            actual_state = self.starting_state
            names_of_possible_actions = self._find_possible_actions(actual_state)
            while names_of_possible_actions is not []:
                self.visited_states.append(actual_state)
                name_of_actual_action = self._choose_action_by_greedy_method(actual_state, names_of_possible_actions)

                next_state = tuple((np.array(actual_state) + self.possible_actions[name_of_actual_action]).tolist())
                names_of_possible_next_actions = self._find_possible_actions(next_state)
                if names_of_possible_next_actions == []:
                    self._update_value_of_last_state_and_action(actual_state, name_of_actual_action)
                    break
                name_of_next_action = self._choose_action_by_greedy_method(next_state, names_of_possible_next_actions)

                self._update_state_action_values(actual_state, name_of_actual_action, next_state, name_of_next_action)

                actual_state = next_state
                name_of_actual_action = name_of_next_action
                names_of_possible_actions = names_of_possible_next_actions
            episode_score[episode_number] = len(self.visited_states)
            self.visited_states = []
        pd.DataFrame.from_dict(episode_score, orient='index', columns=["states_visited"]).to_excel("Summary.xlsx")
        print("The end!")

    def _find_possible_actions(self, actual_state: tuple[int, int]) -> List[str]:
        """Detect all possible actions for actual_state.

        Args:
            actual_state (tuple[int, int]): Actual state we are looking possible actions for.

        Returns:
            List[str]: Returns list of names of possible actions.
        """
        possible_actions = []
        for action_name, action_vector in self.possible_actions.items():
            if self._action_valid(actual_state, action_vector):
                possible_actions.append(action_name)
        return possible_actions

    def _choose_action_by_greedy_method(self, state: tuple[int, int], names_of_possible_actions: List[str]) -> str:
        """Select possible action for a given state by epsilon greedy method.

        Args:
            state (tuple[int, int]): Selected state to pick up action for.
            names_of_possible_actions (List[str]): List of names of possible actions for given state.

        Returns:
            str: Name of selected action.
        """
        name_of_best_action = self._select_best_action(state, names_of_possible_actions)
        if np.random.uniform(0, 1) < self.eps and len(names_of_possible_actions) > 1:
            return self._select_random_not_best_action(name_of_best_action, names_of_possible_actions)
        return name_of_best_action

    def _select_best_action(self, state: tuple[int, int], names_of_possible_actions: List[str]) -> str:
        """Select best action of possible actions for provided state with respect to its Q(state, action) values.

        Args:
            state (tuple[int, int]): State for which we want to find best action.
            names_of_possible_actions (List[str]): List of names of possible actions in selected state.

        Returns:
            str: Name of the best action in provided state.
        """
        all_actions = self.states_actions_values[state]
        possible_actions = {action_name: all_actions[action_name] for action_name in names_of_possible_actions}
        best_action = max(possible_actions, key=possible_actions.get)
        return best_action

    def _select_random_not_best_action(self, name_of_best_action: str, names_of_possible_actions: List[str]) -> str:
        """Select not best action by random.

        Args:
            name_of_best_action (str): Name of best action. This will not be chosen.
            names_of_possible_actions (List[str]): List of names of possible actions in selected state.

        Returns:
            str: Name of not best action in provided state.
        """
        names_of_possible_actions.remove(name_of_best_action)
        return names_of_possible_actions[np.random.randint(0, len(names_of_possible_actions))]

    def _update_value_of_last_state_and_action(self, actual_state: tuple[int, int], name_of_actual_action: str) -> None:
        """Update Q(state, action) values for sub-last state and its action.

        Args:
            actual_state (tuple[int, int]): Actual state.
            name_of_actual_action (str): Name of action in current state.
        """
        self.states_actions_values[actual_state][name_of_actual_action] += self.alpha * self.reward

    def _update_state_action_values(self, actual_state: tuple[int, int], name_of_actual_action: str,
                                    next_state: tuple[int, int], name_of_next_action: str) -> None:
        """Update Q(state, action) values for current state and current action.

        Args:
            actual_state (tuple[int, int]): Actual state.
            name_of_actual_action (str): Name of action in current state.
            next_state (tuple[int, int]): Next state.
            name_of_next_action (str): Name of action in next state.
        """
        self.states_actions_values[actual_state][name_of_actual_action] += self.alpha * (
            self.reward
            + self.gamma * self.states_actions_values[next_state][name_of_next_action]
            - self.states_actions_values[actual_state][name_of_actual_action])


def create_rectangle_chessboard(width: int = 8, height: int = 8) -> List[tuple[int, int]]:
    """Create rectangular chessboard with defined width and height.
    Chessboard is represented as a list of tuples which consist of coordinates (x, y)

    Args:
        width (int, optional): Width of chessboard. Defaults to 8.
        height (int, optional): Height of chessboard. Defaults to 8.

    Returns:
        List[tuple[int, int]]: Created chessboard.
    """
    chessboard = []
    for i in range(0, width):
        for j in range(0, height):
            chessboard.append((i, j))
    return chessboard


def main() -> None:
    chessboard = create_rectangle_chessboard()
    hungry_horse = HungryHorse(chessboard)
    hungry_horse.run_sarsa()
    pass


if __name__ == "__main__":
    main()
