import os
import json
import numpy as np


def timeline_gender(doc_id):
    """Given a doc_id and a gender requirements
    returns intervals to consider """

    # Open json file
    file_name =os.path.join("face-bboxes", doc_id+".json") 
    if not os.path.exists(file_name):
        #print("Not found")
        return {}, {}, {}
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    if not data["faces"]:
        return {}, {}, {}
    
    tmin = int(min([d["t"][0] for d in data["faces"]]))
    tmax = int(np.ceil(max([d["t"][1] for d in data["faces"]])))
    gender = {}
    locations = {}
    persons = {}

    # Create the timeline
    #Create mapping:
    person_mapping = {}
    for name, idx in data["ids"]:
        if idx in person_mapping.keys():
            continue
        person_mapping[idx] = name

    for d in data["faces"]:
        t1 = int(d['t'][0])
        t2 = int(d['t'][1])

        for time_box in range(t1, t2):
            if time_box in gender.keys():
                gender[time_box].append(d['g'])
            else:
                gender[time_box] = [d['g']]

        for time_box in range(t1, t2):
            if time_box in locations.keys():
                locations[time_box].append(d['b'])
            else:
                locations[time_box] = [d['b']]

        for time_box in range(t1, t2):
            if 'i' in d.keys():
                if time_box in persons.keys():
                    persons[time_box].append(person_mapping[d['i']])
                else:
                    persons[time_box] = [person_mapping[d['i']]]


    return gender, locations, persons


def gender_to_time(doc_id, gender_req):
    """Given a doc_id and a gender requirements
    returns intervals to consider """

    # Open json file
    file_name =os.path.join("face-bboxes", doc_id+".json") 
    if not os.path.exists(file_name):
        #print("Not found")
        return []
    with open(file_name, "r") as json_file:
        data = json.load(json_file)

    if not data["faces"]:
        return []
    tmin = int(min([d["t"][0] for d in data["faces"]]))
    tmax = int(np.ceil(max([d["t"][1] for d in data["faces"]])))
    timeline = np.zeros((tmax, 2))

    # Create the timeline
    for d in data["faces"]:
        t1 = int(d['t'][0])
        t2 = int(d['t'][1])
        id_g = 0 if d["g"]=="m" else 1
        timeline[t1:t2, id_g] += 1


    list_suitable_time = list(np.argwhere((timeline[:, 1] >= gender_req["finf"])
           & (timeline[:, 1] <= gender_req["fsup"])
           & (timeline[:, 0] <= gender_req["msup"])
            & (timeline[:, 0] >= gender_req["minf"]))[:, 0])
    
    if not list_suitable_time:
        return []
    intervals = []
    i = 0
    t1 = t = list_suitable_time[0]
    while i < len(list_suitable_time) - 1:
        t1 = t = list_suitable_time[i]
        while list_suitable_time[i+1] == t+1:
            t += 1
            i += 1
            if i >= len(list_suitable_time)-2:
                break
        intervals.append((t1, t))
        i += 1
    intervals.append((t1, t))
    return intervals
