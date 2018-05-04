# Ideal benchmark

- 1 client device
- Uncongested network
    -  Network locked to a mostly unused channel (9)

### Error
Results have one minimal error: there was a bug in the routine for getting the finish timestamp of the runs,
so all the runs have identical init and end timestamps. 

This is minimal, since the task duration can be calculated from the frame timestamps.
