import numpy as np
import pytest
import sys
sys.path.append('../hungry_horse')

from hungry_searching_horse import HungrySearchingHorse
from hungry_searching_horse import Chessboard

horse_parameters_1 = {"chessboard": Chessboard(15, 15, step_cost=-1),
                      "starting_state": (0, 0),
                      "initial_value_for_non_terminal_states": -10,
                      "initial_value_for_terminal_states": 0,
                      "alpha": 0.01,
                      "eps": 0.3,
                      "gamma": 1}
horse_parameters_2 = {"chessboard": Chessboard(3, 3, step_cost=-1),
                      "starting_state": (0, 0),
                      "initial_value_for_non_terminal_states": 0,
                      "initial_value_for_terminal_states": 0,
                      "alpha": 0.01,
                      "eps": 0.3,
                      "gamma": 1}

@pytest.mark.parametrize("horse_parameters, expected_possible_actions",
                         [(horse_parameters_1, {"ul": np.array([-1, 2]),
                                                "ur": np.array([1, 2]),
                                                "ru": np.array([2, 1]),
                                                "rd": np.array([2, -1]),
                                                "dr": np.array([1, -2]),
                                                "dl": np.array([-1, -2]),
                                                "ld": np.array([-2, -1]),
                                                "lu": np.array([-2, 1])})])
def test_generate_possible_actions(horse_parameters, expected_possible_actions):
    horse = HungrySearchingHorse(**horse_parameters)
    np.testing.assert_equal(horse.possible_actions, expected_possible_actions)

@pytest.mark.parametrize("horse_parameters, expected_possible_state_actions",
                         [(horse_parameters_2, {(0, 0): {'ur': 0, 'ru': 0},
                                                (0, 1): {'ru': 0, 'rd': 0},
                                                (0, 2): {'rd': 0, 'dr': 0},
                                                (1, 0): {'ul': 0, 'ur': 0},
                                                (1, 1): {},
                                                (1, 2): {'dr': 0, 'dl': 0},
                                                (2, 0): {'ul': 0, 'lu': 0},
                                                (2, 1): {'ld': 0, 'lu': 0},
                                                (2, 2): {'dl': 0, 'ld': 0}})])
def test_create_combinations_of_all_possible_states_and_actions(horse_parameters, expected_possible_state_actions):
    horse = HungrySearchingHorse(**horse_parameters)
    states_actions = horse._create_combinations_of_all_possible_states_and_actions()
    assert states_actions == expected_possible_state_actions

@pytest.mark.parametrize("horse_parameters, expected_values_of_non_terminal_states",
                         [(horse_parameters_2, {(0, 0): {'ur': 0, 'ru': 0},
                                                (0, 1): {'ru': 0, 'rd': 0},
                                                (0, 2): {'rd': 0, 'dr': 0},
                                                (1, 0): {'ul': 0, 'ur': 0},
                                                (1, 1): {},
                                                (1, 2): {'dr': 0, 'dl': 0},
                                                (2, 0): {'ul': 0, 'lu': 0},
                                                (2, 1): {'ld': 0, 'lu': 0}})])
def test_initialize_non_terminal_states(horse_parameters, expected_values_of_non_terminal_states):
    horse = HungrySearchingHorse(**horse_parameters)
    values_of_non_terminal_states = horse._initialize_non_terminal_states()
    assert values_of_non_terminal_states == expected_values_of_non_terminal_states

@pytest.mark.parametrize("horse_parameters, expected_values_of_terminal_states",
                         [(horse_parameters_2, {(2, 2): {'dl': 0, 'ld': 0}})])
def test_initialize_terminal_state(horse_parameters, expected_values_of_terminal_states):
    horse = HungrySearchingHorse(**horse_parameters)
    values_of_terminal_states = horse._initialize_terminal_state()
    assert values_of_terminal_states == expected_values_of_terminal_states

@pytest.mark.parametrize("horse_parameters, state, action_vector, expected_boolean",
                         [(horse_parameters_2, (0, 0), [2, 1], True),
                          (horse_parameters_2, (0, 0), [-2, 1], False)])
def test_action_valid(horse_parameters, state, action_vector, expected_boolean):
    horse = HungrySearchingHorse(**horse_parameters)
    assert horse._action_valid(state, np.array(action_vector)) == expected_boolean

@pytest.mark.parametrize("horse_parameters, state, expected_possible_actions",
                         [(horse_parameters_2, (0, 0), ["ur", "ru"]),
                          (horse_parameters_2, (2, 2), ["dl", "ld"])])
def test_find_possible_actions(horse_parameters, state, expected_possible_actions):
    horse = HungrySearchingHorse(**horse_parameters)
    assert horse._find_possible_actions(state) == expected_possible_actions

@pytest.mark.parametrize("horse_parameters, state, names_of_possible_actions, expected_chosen_action",
                         [(horse_parameters_2, (0, 0), ["ur", "ru"], "ur"),
                          (horse_parameters_2, (2, 2), ["dl", "ld"], "dl")])
def test_choose_action_by_greedy_method(horse_parameters, state, names_of_possible_actions, expected_chosen_action):
    np.random.seed(0)
    horse = HungrySearchingHorse(**horse_parameters)
    name_of_action = horse._choose_action_by_greedy_method(state, names_of_possible_actions)
    assert name_of_action == expected_chosen_action

@pytest.mark.parametrize("horse_parameters, state, names_of_possible_actions, expected_best_action",
                         [(horse_parameters_2, (0, 0), ["ur", "ru"], "ur"),
                          (horse_parameters_2, (2, 2), ["dl", "ld"], "dl")])
def test_select_best_action(horse_parameters, state, names_of_possible_actions, expected_best_action):
    np.random.seed(0)
    horse = HungrySearchingHorse(**horse_parameters)
    name_of_best_action = horse._select_best_action(state, names_of_possible_actions)
    assert name_of_best_action == expected_best_action

@pytest.mark.parametrize("horse_parameters, name_of_best_action, names_of_possible_actions, \
                         expected_chosen_not_best_action",
                         [(horse_parameters_2, "ru", ["ur", "ru"], "ur"),
                          (horse_parameters_2, "ld", ["dl", "ld"], "dl")])
def test_select_random_not_best_action(horse_parameters, name_of_best_action, names_of_possible_actions,
                                       expected_chosen_not_best_action):
    np.random.seed(0)
    horse = HungrySearchingHorse(**horse_parameters)
    name_of_chosen_not_best_action = horse._select_random_not_best_action(name_of_best_action,
                                                                          names_of_possible_actions)
    assert name_of_chosen_not_best_action == expected_chosen_not_best_action

@pytest.mark.parametrize("horse_parameters, current_state, name_of_current_action, expected_next_state",
                         [(horse_parameters_2, (0, 0), "ru", (2, 1)),
                          (horse_parameters_2, (2, 2), "ld", (0, 1))])
def test_calc_next_state(horse_parameters, current_state, name_of_current_action, expected_next_state):
    horse = HungrySearchingHorse(**horse_parameters)
    next_state = horse._calc_next_state(current_state, name_of_current_action)
    assert next_state == expected_next_state

@pytest.mark.parametrize("horse_parameters, current_state, name_of_current_action, next_state, name_of_next_state,\
                         expected_q",
                         [(horse_parameters_1, (2, 2), "ld", (0, 1), "ur", -10.01),
                          (horse_parameters_2, (0, 0), "ru", (2, 1), "ld", -0.01)])
def test_update_state_action_values(horse_parameters, current_state, name_of_current_action, next_state,
                                    name_of_next_state, expected_q):
    horse = HungrySearchingHorse(**horse_parameters)
    horse._update_state_action_values(current_state, name_of_current_action, next_state, name_of_next_state)
    assert horse.states_actions_values[current_state][name_of_current_action] == expected_q
