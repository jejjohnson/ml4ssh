def add_winter_smoke_test_args(args):
    # change subset data (process)
    args.time_min = "2016-12-01"
    args.time_max = "2017-03-01"
    
    # change subset data (postprocess)
    args.eval_time_min = "2016-12-01"
    args.eval_time_max = "2017-03-01"
    return args

def add_summer_smoke_test_args(args):
    # change subset data (process)
    args.time_min = "2017-06-01"
    args.time_max = "2017-09-01"
    
    # change subset data (postprocess)
    args.eval_time_min = "2017-06-01"
    args.eval_time_max = "2017-09-01"
    return args

def add_january_smoke_test_args(args):
    # change subset data (process)
    args.time_min = "2017-01-01"
    args.time_max = "2017-02-01"
    
    # change subset data (postprocess)
    args.eval_time_min = "2017-01-01"
    args.eval_time_max = "2017-02-01"
    return args

def add_july_smoke_test_args(args):
    # change subset data (process)
    args.time_min = "2017-07-01"
    args.time_max = "2017-08-01"
    
    # change subset data (postprocess)
    args.eval_time_min = "2017-07-01"
    args.eval_time_max = "2017-08-01"
    return args