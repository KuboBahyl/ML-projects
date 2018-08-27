from pprint import pprint
from utils import *

if __name__ == '__main__':
    '''
    Pipeline description:
        - after loading and preprocessing datetimes, the flights are stored as Pandas DataFrame
        (Pandas could be replaced by Spark or other distributed form with big data volume)
        - recursive chronological search within flights, creating a tree structure with origin
        airports as the main key. Then each tree leg represent a possible transfer with implemented
        logic about bags and stopover time. Such structure is scalable with future flights and
        could be saved as json
        - search this tree to make all flights combinations
        - filter out the possible duplicates and cycles (repeated routes)
        - add price as the main attribute to each combination
        - print combinations in a nice way - for each bags option as a list of lists
        - sample combination list consists flight_numbers with total price at the end

    Comments:
        - run code as `cat flights.csv | python find_combinations.py`
        - most of functions assume that flight_numbers are unique among dataset
        - function for loading data is not optimised for real-time streaming, it just waits
        for the static stdinput
        - tree searching is limited to max 2 stopovers
        - if interested in searching combinations from certain airport, it can be
        easily specified before tree searching
    '''
    flights = stdin_flights()
    flights = preprocess(flights)

    # 0,1,2 bags taken
    for bags in range(3):
        flights_tree = make_tree(flights, num_bags=bags)
        combinations = search_combinations(flights_tree)
        combinations = filter_cycles(combinations, flights)
        combinations = filter_duplicates(combinations)
        combinations = add_prices(combinations, flights, num_bags=bags)
        print("\n Flights combinations and prices for number of bags: {}".format(bags))
        pprint(combinations)
