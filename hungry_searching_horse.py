from typing import Dict, List, Union

import pandas as pd
import pickle
import numpy as np

from chessboard import Chessboard
from PIL import Image


class HungrySearchingHorse:

    def __init__(self,
                 chessboard: Chessboard,
                 starting_state: tuple[int, int] = (0, 0),
                 initial_value_for_non_terminal_states: int = 0,
                 initial_value_for_terminal_states: int = 0,
                 alpha: float = 0.01,
                 eps: float = 0.3,
                 gamma: float = 1) -> None:
        """
        Args:
            chessboard (Chessboard): Instance of a class Chessboard.
            starting_state (tuple[int, int], optional): Starting position of a horse. Defaults to (0, 0).
            initial_value_for_non_terminal_states (int, optional): Initial value of Q(state, action) for non terminal
                                                                   states. Defaults to 0.
            initial_value_for_terminal_states (int, optional): Initial value of Q(state, action) for terminal states.
                                    Should be higher or equal to initial_value_for_non_terminal_states. Defaults to 0.
            alpha (float, optional): Gradient step parameter. Defaults to 0.1.
            eps (float, optional): Parameter for epsilon greedy search. Defaults to 0.05.
            gamma (float, optional): Factor to multiply Q(state, value) of next state-action pair. Defaults to 0.9.
        """
        self.chessboard = chessboard
        self.starting_state = starting_state
        self.terminal_state = (self.chessboard.width-1, self.chessboard.height-1)
        self.possible_actions = self._generate_possible_actions()
        self.initial_value_for_non_terminal_states = initial_value_for_non_terminal_states
        self.initial_value_for_terminal_states = initial_value_for_terminal_states
        self.states_actions_values = self._create_combinations_of_all_possible_states_and_actions()

        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma

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
        """Initialize Q(state, value) pairs.

        Returns:
            Dict[tuple[int, int], Dict[str, float]]: All Q(state, action) values initialized to be zero.
        """
        non_terminal_states_actions_values = self._initialize_non_terminal_states()
        terminal_states_actions_values = self._initialize_terminal_state()
        return {**non_terminal_states_actions_values, **terminal_states_actions_values}

    def _initialize_non_terminal_states(self) -> Dict[tuple[int, int], Dict[str, float]]:
        """Create combinations of all possible non terminal states and actions. Results are stored as a Dict of
        dictionaries. The first key is meant to be coordinates of actual state. The value is inner dictionary where key
        is name of action and value is its Q(state, action) value initialized to be zero.

        Returns:
            Dict[tuple[int, int], Dict[str, float]]: All non terminal Q(state, action) values initialized to be
                                                    self.initial_value_for_non_terminal_states.
        """
        states_actions_values = {}
        non_terminal_states = list(self.chessboard.states_rewards.keys())
        non_terminal_states.remove(self.terminal_state)
        for state in non_terminal_states:
            valid_actions = {}
            for action_name, action_vector in list(self.possible_actions.items()):
                if self._action_valid(state, action_vector):
                    valid_actions[action_name] = self.initial_value_for_non_terminal_states
            states_actions_values[state] = valid_actions
        return states_actions_values

    def _initialize_terminal_state(self) -> Dict[tuple[int, int], Dict[str, float]]:
        """Initialize Q(state, action) values for terminal state and all its possible actions to be equal to
        self.initial_value_for_terminal_states.

        Returns:
            Dict[tuple[int, int], Dict[str, float]]: Dictionary of all Q(state, value) pairs and their initial values.
        """
        valid_actions = {}
        for action_name, action_vector in list(self.possible_actions.items()):
            if self._action_valid(self.terminal_state, action_vector):
                valid_actions[action_name] = self.initial_value_for_terminal_states
        terminal_states_actions_values = {self.terminal_state: valid_actions}
        return terminal_states_actions_values

    def _action_valid(self, state: tuple[int, int], action_vector: np.ndarray[int, int]) -> bool:
        """Check if action can be done in provided state.

        Args:
            state (tuple[int, int]): Starting state on a chessboard.
            action_vector (np.ndarray[int, int]): Vector of movement for selected action.

        Returns:
            bool: If action is valid return True else False.
        """
        next_state = tuple((np.array(state) + action_vector).tolist())
        if next_state in list(self.chessboard.states_rewards.keys()):
            return True
        return False

    def run_q_learning(self, amount_of_episodes: int = 10_000) -> None:
        """
        Run Q learning algorithm to find solution.
        Optimized values of Q(state, action) pairs are stored in pickle file state_action_values_q_learning.pickle.
        Summary about total score and amount of steps in every episode is stored in excel file Summary_q_learning.xlsx.

        Args:
            amount_of_episodes (int, optional): Maximum amount of episodes used for optimization. Defaults to 10_000.
        """
        episode_score_and_steps = {}
        for episode_number in range(0, amount_of_episodes):
            print("episode number " + str(episode_number))
            actual_state = self.starting_state
            names_of_possible_actions = self._find_possible_actions(actual_state)
            step = 0
            score = 0
            while actual_state != self.terminal_state:
                name_of_actual_action = self._choose_action_by_greedy_method(actual_state, names_of_possible_actions)
                next_state = self._calc_next_state(actual_state, name_of_actual_action)
                names_of_possible_next_actions = self._find_possible_actions(next_state)
                name_of_next_action = self._select_best_action(next_state, names_of_possible_next_actions)

                score += self.chessboard.states_rewards[next_state]
                step += 1

                self._update_state_action_values(actual_state, name_of_actual_action, next_state, name_of_next_action)

                actual_state = next_state
                name_of_actual_action = name_of_next_action
                names_of_possible_actions = names_of_possible_next_actions

        episode_score_and_steps[episode_number] = [score, step]

        with open('state_action_values_q_learning.pickle', 'wb') as handle:
            pickle.dump(self.states_actions_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame.from_dict(episode_score_and_steps,
                               orient='index',
                               columns=["Total_score", "Steps_in_episode"]).to_excel("Summary_q_learning.xlsx")

    def run_sarsa(self, amount_of_episodes: int = 10_000) -> None:
        """
        Run SARSA algorithm to find solution.
        Optimized values of Q(state, action) pairs are stored in pickle file state_action_values_sarsa.pickle.
        Summary about total score and amount of steps in every episode is stored in excel file Summary_SARSA.xlsx.

        Args:
            amount_of_episodes (int, optional): Maximum amount of episodes used for optimization. Defaults to 10_000.
        """
        episode_score_and_steps = {}
        for episode_number in range(0, amount_of_episodes):
            print("episode number " + str(episode_number))
            actual_state = self.starting_state
            names_of_possible_actions = self._find_possible_actions(actual_state)
            step = 0
            score = 0
            while actual_state != self.terminal_state:
                name_of_actual_action = self._choose_action_by_greedy_method(actual_state, names_of_possible_actions)
                next_state = self._calc_next_state(actual_state, name_of_actual_action)
                names_of_possible_next_actions = self._find_possible_actions(next_state)
                name_of_next_action = self._choose_action_by_greedy_method(next_state, names_of_possible_next_actions)

                score += self.chessboard.states_rewards[next_state]
                step += 1

                self._update_state_action_values(actual_state, name_of_actual_action, next_state, name_of_next_action)

                actual_state = next_state
                name_of_actual_action = name_of_next_action
                names_of_possible_actions = names_of_possible_next_actions

        episode_score_and_steps[episode_number] = [score, step]

        with open('state_action_values_sarsa.pickle', 'wb') as handle:
            pickle.dump(self.states_actions_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame.from_dict(episode_score_and_steps,
                               orient='index',
                               columns=["Total_score", "Steps_in_episode"]).to_excel("Summary_SARSA.xlsx")

    def run_n_step_sarsa(self, amount_of_steps: int, amount_of_episodes: int = 10_000) -> None:
        """Run n_step_SARSA algorithm to find solution.
        Optimized values of Q(state, action) pairs are stored in pickle file state_action_values_n_step_sarsa.pickle.
        Summary about total score and amount of steps in every episode is stored in excel file
        Summary_n_step_SARSA.xlsx.

        Args:
            amount_of_steps (int): Amount of steps to use for estimation of gain in current state.
            amount_of_episodes (int, optional): Maximum amount of episodes used for optimization. Defaults to 10_000.
        """
        episode_score_and_steps = {}
        for episode_number in range(0, amount_of_episodes):
            print("episode number " + str(episode_number))
            starting_state = self._initialize_starting_state()
            ending_time = np.inf  # T
            tau = None
            step = 0
            score = 0
            actual_time_step = 0  # t
            sequence_of_rewards = []
            sequence_of_visited_states = [starting_state]
            names_of_possible_actions = self._find_possible_actions(sequence_of_visited_states[-1])
            name_of_actual_action = self._choose_action_by_greedy_method(sequence_of_visited_states[-1],
                                                                         names_of_possible_actions)
            sequence_of_actions = [name_of_actual_action]
            while tau != ending_time - 1:
                if actual_time_step < ending_time:
                    next_state = self._calc_next_state(sequence_of_visited_states[-1], sequence_of_actions[-1])
                    sequence_of_rewards.append(self.chessboard.states_rewards[next_state])
                    sequence_of_visited_states.append(next_state)

                    score += self.chessboard.states_rewards[next_state]
                    step += 1

                    if next_state == self.terminal_state:
                        ending_time = actual_time_step + 1
                    else:
                        names_of_possible_next_actions = self._find_possible_actions(sequence_of_visited_states[-1])
                        name_of_next_action = self._choose_action_by_greedy_method(sequence_of_visited_states[-1],
                                                                                   names_of_possible_next_actions)
                        sequence_of_actions.append(name_of_next_action)

                tau = actual_time_step - amount_of_steps + 1
                if tau >= 0:
                    total_gain = 0
                    for i in range(tau+1, min(tau+amount_of_steps+1, ending_time)):
                        total_gain += self.gamma**(i-tau-1)*sequence_of_rewards[i-1]
                    if tau + amount_of_steps < ending_time:
                        total_gain = (total_gain
                                      + self.gamma**amount_of_steps
                                      * self.states_actions_values[sequence_of_visited_states[tau + amount_of_steps]][
                                        sequence_of_actions[tau + amount_of_steps]])
                    self.states_actions_values[sequence_of_visited_states[tau]][sequence_of_actions[tau]] += (
                            self.alpha
                            * (self.gamma*total_gain
                            - self.states_actions_values[sequence_of_visited_states[tau]][sequence_of_actions[tau]]))
                actual_time_step += 1
            episode_score_and_steps[episode_number] = [score, step]
        with open('state_action_values_n_step_sarsa.pickle', 'wb') as handle:
            pickle.dump(self.states_actions_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame.from_dict(episode_score_and_steps,
                               orient='index',
                               columns=["Total_score", "Steps_in_episode"]).to_excel("Summary_n_step_SARSA.xlsx")

    def _initialize_starting_state(self) -> tuple[int, int]:
        """Initialize starting state randomly. Must not be equal to the terminal state.

        Returns:
            tuple[int, int]: Coordinates of starting state.
        """
        while True:
            selected_starting_state = list(self.states_actions_values.keys())[
                                np.random.randint(0, len(list(self.states_actions_values.keys())))]
            if selected_starting_state != self.terminal_state:
                return selected_starting_state

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

    def _calc_next_state(self, current_state: tuple[int, int], name_of_current_action: str) -> tuple[int, int]:
        """Calculate position of next state.

        Args:
            current_state (tuple[int, int]): Coordinates of current state.
            name_of_current_action (str): Name of current action taken in current state.

        Returns:
            tuple[int, int]: Coordinates of next state.
        """
        return tuple((np.array(current_state) + self.possible_actions[name_of_current_action]).tolist())

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
            self.chessboard.states_rewards[next_state]
            + self.gamma * self.states_actions_values[next_state][name_of_next_action]
            - self.states_actions_values[actual_state][name_of_actual_action])

    def evaluate_solution(self) -> Union[float, int]:
        """Method to evaluate solution according to Q(state, action) pair values.

        Returns:
            float: Total score of the best solution.
            int: Amount of steps for the best solution.
        """
        actual_state = self.starting_state
        name_of_actual_action = self._select_best_action(self.starting_state,
                                                         list(self.states_actions_values[self.starting_state].keys()))
        step = 0
        score = 0
        while actual_state != self.terminal_state:
            next_state = self._calc_next_state(actual_state, name_of_actual_action)
            score += self.chessboard.states_rewards[next_state]
            actual_state = next_state
            name_of_actual_action = self._select_best_action(actual_state,
                                                             list(self.states_actions_values[actual_state].keys()))
            step += 1
        return score, step

    def create_animation_of_solution(self) -> None:
        """Iterate over dictionary of Q(state, action) values pair to find the best path.
        Every step is saved as jpg into folder.
        """
        img_of_users_chessboard = self.chessboard.img_of_modified_chessboard
        pixels = img_of_users_chessboard.load()
        color_of_actual_state = (255, 128, 0)  # Orange
        color_of_next_state = (178, 102, 255)  # Purple
        pixels[self.starting_state] = color_of_actual_state

        name_of_actual_action = self._select_best_action(self.starting_state,
                                                         list(self.states_actions_values[self.starting_state].keys()))
        actual_state = self.starting_state
        step = 0
        while actual_state != self.terminal_state:
            next_state = self._calc_next_state(actual_state, name_of_actual_action)

            pixels[actual_state] = color_of_actual_state
            pixels[next_state] = color_of_next_state
            img_of_users_chessboard.resize(
                size=(300, 300),
                resample=Image.NEAREST).transpose(Image.FLIP_TOP_BOTTOM).save("Chessboard_"+str(step)+".jpg")

            actual_state = next_state
            name_of_actual_action = self._select_best_action(actual_state,
                                                             list(self.states_actions_values[actual_state].keys()))
            step += 1


def main() -> None:
    chessboard = Chessboard(15, 15)
    chessboard.modify_chessboard()
    chessboard.draw_users_chessboard()

    hungry_horse = HungrySearchingHorse(chessboard)
    hungry_horse.run_sarsa(20_000)
    hungry_horse.create_animation_of_solution()


if __name__ == "__main__":
    np.random.seed(0)
    main()
    print("Script run successfully!")
