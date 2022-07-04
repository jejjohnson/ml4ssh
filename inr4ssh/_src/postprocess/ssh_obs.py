def postprocess(ds_oi, ds_correct):



    # interpolate the points to evaluation grid
    ds_correct = ds_correct.interp(longitude=ds_oi.longitude, latitude=ds_oi.latitude)

    # add correction
    ds_oi["ssh"] = ds_oi["ssh"] + ds_correct["mdt"]

    return ds_oi