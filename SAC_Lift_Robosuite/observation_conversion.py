def obs_convert(obs):
    obs_list = []
    for _, value in obs.items():
        obs_list.extend(value.tolist())  # extend, will avoid nested list formation and make a single list
    return obs_list