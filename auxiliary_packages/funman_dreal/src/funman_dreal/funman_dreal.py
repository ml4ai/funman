import os
import docker
from shutil import copyfile, rmtree
import time


# def setup(smt2_file, benchmark_path, out_dir):
#     print("setting up docker")
#     if os.path.exists(out_dir):
#         rmtree(out_dir)
#     os.makedirs(out_dir)
#     copyfile(
#         os.path.join(benchmark_path, smt2_file),
#         os.path.join(out_dir, smt2_file),
#     )


# TODO find a better way to determine if solver was successful
def run_dreal(
    smt2_file,
    # benchmark_path=os.path.join(os.path.abspath("../../resources/smt2")),
    out_dir="out",
    solver_opts=" --precision 1 --verbose debug",
):
    # setup(smt2_file, benchmark_path, out_dir)

    smt2_file_path = os.path.dirname(smt2_file)
    smt2_filename = os.path.basename(smt2_file)

    volumes = {
        smt2_file_path: {"bind": "/smt2", "mode": "rw"},
        out_dir: {"bind": "/out", "mode": "rw"},
        "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
    }
    client = docker.from_env()
    container = client.containers.run(
        "dreal/dreal4",
        # command=f"dreal /smt2/{smt2_filename} {solver_opts}",
        command="/bin/bash",
        volumes=volumes,
        detach=True,
    )
    container.wait()
    result = container.logs()

    with open(os.path.join(out_dir, f"{smt2_filename}.stdout"), "w") as f:
        f.write(result.decode())
        f.close()

    if not os.path.exists(os.path.join(out_dir, f"{smt2_filename}.model")):
        container.stop()
        ll_client = docker.APIClient()
        ll_client.remove_container(container.id)
        raise Exception("Could not construct a model")

    container.stop()
    ll_client = docker.APIClient()
    ll_client.remove_container(container.id)

    with open(os.path.join(out_dir, f"{smt2_filename}.model")) as f:
        model = f.read()

    return model
