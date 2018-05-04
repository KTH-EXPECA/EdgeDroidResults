# Ideal benchmark

- 1 client device
- Uncongested network
    -  Network locked to a mostly unused channel (9)
- 20MHz band
- 2.4Ghz, 802.11n

### Error
Results have one minimal error: there was a bug in the routine for getting the finish timestamp of the runs,
so all the runs have identical init and end timestamps. 

This is minimal, since the task duration can be calculated from the frame timestamps.
