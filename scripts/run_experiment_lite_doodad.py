import doodad
def failure():
    raise ValueError("Must provide run_method via doodad args!")
fn = doodad.get_args('run_method', failure)
fn()
