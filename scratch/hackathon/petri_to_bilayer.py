import json


def petri_to_bilayer(petri_net_path):
    """Takes in a LabelledPetriNet JSON object and outputs a BiLayer JSON object"""
    ### Initialize lists to build outputs
    Wa_list = []
    Wn_list = []
    Wa_index_list = []
    Wn_index_list = []
    Win_list = []
    Box_list = []
    Qin_list = []
    Qout_list = []
    ### Load data from path
    f = open(petri_net_path)
    petri_net_src = json.load(f)
    ### Get variables in Qin and tanvars in Qout
    i = 1
    for state_var in petri_net_src["S"]:
        state_var["index"] = i
        i += 1
        state_var_name = state_var["sname"]
        Qin_list.append({"variable": f"{state_var_name}"})
        Qout_list.append({"tanvar": f"{state_var_name}'"})
    #    print(Qin_list)
    #    print(Qout_list)
    ### Get parameters in Box
    i = 1
    for param in petri_net_src["T"]:
        param["index"] = i
        i += 1
        param_name = param["tname"]
        Box_list.append({"parameter": f"{param_name}"})
    #    print(Box_list)
    ### Get Wa
    i = 1
    for outtransition in petri_net_src["O"]:
        outtransition["index"] = i
        i += 1
        ot = outtransition["ot"]
        os = outtransition["os"]
        Wa_index_list.append((ot, os))
    #        Wa_list.append({"influx": ot, "infusion": os})
    #    print(Wa_list)
    ### Get Wn
    i = 1
    for intransition in petri_net_src["I"]:
        intransition["index"] = i
        i += 1
        it = intransition["it"]
        iss = intransition["is"]
        Wn_index_list.append((it, iss))
        Win_list.append({"arg": iss, "call": it})
    #        Wn_list.append({"efflux": it, "effusion": iss})
    #    print(Wn_list)
    ########## Optional: getting rid of redundant edges
    for elt in Wa_index_list:
        if elt in Wn_index_list:
            Wa_index_list.remove(elt)
            Wn_index_list.remove(elt)
    for elt in Wa_index_list:
        Wa_list.append({"influx": elt[0], "infusion": elt[1]})
    for elt in Wn_index_list:
        Wn_list.append({"efflux": elt[0], "effusion": elt[1]})
    ###################################################
    output = {
        "Qin": Qin_list,
        "Qout": Qout_list,
        "Box": Box_list,
        "Wa": Wa_list,
        "Wn": Wn_list,
        "Win": Win_list,
    }
    json_output = json.dumps(output, indent=4)
    return json_output


################ Example use case

print(petri_to_bilayer("../resources/petrinet/SIDARTHE_petri_DMI.json"))
