## Overview

This repository contains a solution of Kiwi Python Weekend task - a searching of flight combinations with given data.

The solution can be replicated with following call from Terminal:

`cat flights.csv | python find_combinations.py`

It is expected to have installed Python 3 and Pandas library. The output is formatted with a following schema:

```
[
  ['PV123', 'PV456', 100],
  ['PV789', 50],
  ...
]
```

, where `PVXXX` is a flight number and the last number in each nested list is the total price. This is printed 3-times, for baggage cases: 0, 1 and 2 pieces.

## Implemented logic

The code is written in a way that it assumes:

* no NaNs within dataset, stable format
* static input, no stream handling
* flight numbers are unique
* 1 - 4 hours for stopover between flights
* no duplicated routes (oriented connection from A to B)
