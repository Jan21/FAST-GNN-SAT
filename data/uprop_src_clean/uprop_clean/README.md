# uprop
Simple implementation of unit propagation for conjunctive normal form (CNF).

# building

On a Linux machine, the `configure` script should download minisat and/or cadical
and prepare the `build` folder.

Run `./configure -h`  to see options for building.

In a nutshell, it should be enough to do:
```
     ./configure && cd build && make
```

