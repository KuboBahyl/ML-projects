import csv
import sys
import pandas as pd
from datetime import datetime
from time import mktime

# Data loading from stdin
def stdin_flights() -> pd.DataFrame:
    f = sys.stdin.read().splitlines()
    lines = list(csv.reader(f))
    column_names = lines[0]
    data = lines[1:]
    num_rows = len(data)

    # filling DF with data
    flights = pd.DataFrame(columns=column_names)
    for i in range(num_rows):
        flights.loc[i] = data[i]

    # convert to numeric cols
    numeric_columns = ['price', 'bags_allowed', 'bag_price']
    flights[numeric_columns] = flights[numeric_columns].apply(pd.to_numeric)

    return flights

# Preprocessing
def date2timestamp(date: str):
    return mktime(datetime.strptime(date, "%Y-%m-%dT%H:%M:%S").timetuple())

def preprocess(flights: pd.DataFrame):
    flights['departure'] = flights['departure'].apply(lambda x: date2timestamp(x))
    flights['arrival'] = flights['arrival'].apply(lambda x: date2timestamp(x))
    return flights

# search for flights combination
def make_tree(flights: pd.DataFrame, num_bags=0):
    tree = {}
    airports = flights['source'].unique()

    # search for flights within subtree
    def make_subtree(flight: str, flights: pd.DataFrame):
        sub_tree = {}
        row = flights[flights['flight_number'] == flight]
        time_now, airport_now = row[['arrival', 'destination']].values[0]
        flights_future = flights[flights['departure'] >= time_now + 60*60]
        flights_now = flights_future[flights_future['departure'] <= time_now + 4*60*60]

        if len(flights_now) == 0 or airport_now not in flights_now['source'].values:
            return sub_tree
        else:
            airport_flights_now = flights_now['flight_number'][ flights_now['source'] == airport_now ]
            for flight_now in airport_flights_now:
                sub_tree[flight_now] = make_subtree(flight_now, flights_future)

            return sub_tree


    for airport in airports:
        tree[airport] = {}
        airport_flights = flights['flight_number'][(flights['source'] == airport) & \
                                                   (flights['bags_allowed'] >= num_bags)]

        for flight in airport_flights:
            sub_tree = make_subtree(flight, flights)
            tree[airport][flight] = sub_tree

    return tree

# assuming there are no more than 2 stopovers
def search_combinations(tree: dict):
    combinations_all = []
    for airport in tree:
        for flight in tree[airport]:
            combination = [flight]
            combinations_all += [combination]

            for flight_next in tree[airport][flight]:
                combination_next = [flight_next]
                combinations_all += [combination_next]
                combinations_all += [combination + combination_next]

                for flight_next_next in tree[airport][flight][flight_next]:
                    combination_next_next = [flight_next_next]
                    combinations_all += [combination_next_next]
                    combinations_all += [combination_next + combination_next_next]
                    combinations_all += [combination + combination_next + combination_next_next]

    return combinations_all

# remove the cases like A -> B -> A -> B
def filter_cycles(combinations: list, flights: pd.DataFrame):
    for combination in combinations:
        if len(combination) >= 3: # able to make cycled route
            routes = []
            for flight in combination:
                row = flights[flights['flight_number'] == flight]
                route = row[['source', 'destination']].values[0]
                if list(route) in routes:
                    combinations.remove(combination)
                    break
                else:
                    routes += [list(route)]

    return combinations

# remove the same flight combinations
def filter_duplicates(combinations: list):
    combinations_unique = set(tuple(x) for x in combinations)
    return [list(x) for x in combinations_unique]

# calculate price for each combination list
def add_prices(combinations: list, flights: pd.DataFrame, num_bags=0):
    num_combination = len(combinations)
    for i in range(num_combination):
        combination_price = 0
        for flight in combinations[i]:
            row = flights[flights['flight_number'] == flight]
            combination_price += row['price'].values[0] + num_bags * row['bag_price'].values[0]

        combinations[i] += [combination_price]

    return combinations
