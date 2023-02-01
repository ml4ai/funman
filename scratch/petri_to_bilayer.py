import json

def petri_to_bilayer(petri_net_path):
    """Takes in a LabelledPetriNet JSON object and outputs a BiLayer JSON object"""
    ### Initialize lists to build outputs
    Wa_list = []
    Wn_list = []
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
        Wa_list.append({"influx": ot, "infusion": os})
    #    print(Wa_list)
    ### Get Wn
    i = 1
    for intransition in petri_net_src["I"]:
        intransition["index"] = i
        i += 1
        it = intransition["it"]
        iss = intransition["is"]
        Wn_list.append({"efflux": it, "effusion": iss})
    #    print(Wn_list)
    ### Get Win
    for state_var in petri_net_src["S"]:
        for out_of_transition_edge in petri_net_src[
            "O"
        ]:  ## Loop over edges that will be added
            if out_of_transition_edge["os"] == state_var["index"]:
                summand = out_of_transition_edge[
                    "ot"
                ]  ## Finding relevant transition
                for transition in petri_net_src["T"]:
                    if transition["index"] == summand:
                        ### Find state vars that point into transition
                        for into_transition_edge in petri_net_src["I"]:
                            if (
                                into_transition_edge["it"]
                                == transition["index"]
                            ):
                                ### Look up state variable with index "is"
                                state_variable_src_index = (
                                    into_transition_edge["is"]
                                )
                                for state_var in petri_net_src["S"]:
                                    if (
                                        state_var["index"]
                                        == state_variable_src_index
                                    ):
                                        Win_list_input = {
                                            "arg": state_var["index"],
                                            "call": transition["index"],
                                        }
                                        if Win_list_input not in Win_list:
                                            Win_list.append(Win_list_input)
        for into_transition_edge in petri_net_src["I"]:
            if into_transition_edge["is"] == state_var["index"]:
                summand = into_transition_edge[
                    "it"
                ]  ## Finding relevant transition
                for transition in petri_net_src["T"]:
                    if transition["index"] == summand:
                        ### Find state vars that point into transition
                        for into_transition_edge in petri_net_src["I"]:
                            if (
                                into_transition_edge["it"]
                                == transition["index"]
                            ):
                                ### Look up state variable with index "is"
                                state_variable_src_index = (
                                    into_transition_edge["is"]
                                )
                                for state_var in petri_net_src["S"]:
                                    if (
                                        state_var["index"]
                                        == state_variable_src_index
                                    ):
                                        Win_list_input = {
                                            "arg": state_var["index"],
                                            "call": transition["index"],
                                        }
                                        if Win_list_input not in Win_list:
                                            Win_list.append(Win_list_input)
    #    print(Win_list) ### actually want to take the set of this
    output = {
        "Qin": Qin_list,
        "Qout": Qout_list,
        "Box": Box_list,
        "Wa": Wa_list,
        "Wn": Wn_list,
        "Win": Win_list,
    }
    json_output = json.dumps(output, indent = 4)
    return json_output


################ Example use case
print(
    petri_to_bilayer("../resources/petrinet/CHIME_SIR_dynamics_PetriNet.json")
)
