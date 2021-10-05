import datetime
import numpy as np
import pandas as pd

def storm_def(data, mit, dt_col, p_col):
        
    numtotal = data.shape[0]

    # Define minimum interevent time (MIT), hours:
    # mit = 5

    # Initialize last heard tracker & rain event counter:
    lastheard = data[dt_col][0] - datetime.timedelta(hours=mit+1)
    rainevent = 1
    Event = dict()

    # Filter rainfalls and create storm event objects:
    for i in range(1, numtotal):
        Event[rainevent] = dict()
        Event[rainevent][dt_col] = []
        Event[rainevent][p_col] = []

        if data[p_col][i] > 0 and data[p_col][i-1] == 0:
            # This is the start of a rain event!

            # Save the start time for the storm event:
            stormstart = data[dt_col][i]

            # Check to see if the MIT requirement is met:
            delta_t = (stormstart - lastheard).total_seconds()/3600.0

            if delta_t <= mit:
                # Found a short break in the storm; still in previous event:
                rainevent -= 1

                # Include the starting zero event:
                tprev = data[dt_col][i-1]
                rprev = data[p_col][i-1]
                if  Event[rainevent][dt_col][-1] != tprev:
                    Event[rainevent][dt_col].append(data[dt_col][i-1])
                    Event[rainevent][p_col].append(data[p_col][i-1])
                    

                # Initialize the current rain events rainfall amount:
                eventrain = data[p_col][i]

                # Initialize the iterater:
                j = 0

                while (eventrain > 0):
                    # Save the time and rain amounts for the event:
                    Event[rainevent][dt_col].append(data[dt_col][i+j])
                    Event[rainevent][p_col].append(data[p_col][i+j])

                    # Increment the iterater & update event rainfall:
                    j += 1
                    eventrain = data[p_col][i+j]

                # Include the ending zero event:
                Event[rainevent][dt_col].append(data[dt_col][i+j])
                Event[rainevent][p_col].append(data[p_col][i+j])

                # Update lastheard date:
                lastheard = Event[rainevent][dt_col][-1]

                # Increment the event couter:
                rainevent += 1

            else:
                # if args.verbose:
                #     print("Start rain event %d: %s, last heard %0.3f hrs" % (
                #         rainevent, stormstart, delta_t))

                # Initialize event object arrays:
                # Note, set the rainfall rate boolean here
                

                # Include the starting zero event:
                Event[rainevent][dt_col].append(data[dt_col][i-1])
                Event[rainevent][p_col].append(data[p_col][i-1])

                # Initialize the current rain events rainfall amount:
                eventrain = data[p_col][i]

                # Initialize the iterater:
                j = 0

                while (eventrain > 0):
                    # Save the time and rain amounts for the event:
                    Event[rainevent][dt_col].append(data[dt_col][i+j])
                    Event[rainevent][p_col].append(data[p_col][i+j])

                    # Increment the iterater & update event rainfall:
                    j += 1
                    eventrain = data[p_col][i+j]

                # Include the ending zero event:
                Event[rainevent][dt_col].append(data[dt_col][i+j])
                Event[rainevent][p_col].append(data[p_col][i+j])
                    

                # Update lastheard date:
                lastheard = Event[rainevent][dt_col][-1]

                # Increment the event couter:
                rainevent += 1
    p_events = []
    for i in range(1,len(Event)):
        p_events.append((Event[i][dt_col][0], Event[i][dt_col][-1], np.sum(Event[i][p_col])/((Event[i][dt_col][-1] - Event[i][dt_col][0]).total_seconds()/3600.0), (Event[i][dt_col][-1] - Event[i][dt_col][0]).total_seconds()/3600.0))
    p_events = np.array(p_events)
    p_events = pd.DataFrame(p_events, columns=['start', 'end', 'precip', 'deltat'])
    return p_events