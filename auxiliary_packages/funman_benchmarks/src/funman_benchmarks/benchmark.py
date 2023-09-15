import datetime
import json
import os
import unittest
from contextlib import contextmanager
from functools import partial
from timeit import default_timer

from interruptingcow import timeout


class Benchmark(unittest.TestCase):
    RESOURCES = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../resources"
    )

    @contextmanager
    def elapsed_timer(self):
        start = default_timer()
        elapser = lambda: default_timer() - start
        try:
            yield elapser
        finally:
            elapser = None

    def case_already_ran(self, case, scenario):
        if os.path.exists(self.out_file):
            with open(self.out_file, "r") as f:
                try:
                    json_results = json.loads(f.read())
                    for result in json_results["results"]:

                        matches = all(
                            [
                                (k in result and result[k] == v)
                                for k, v in case.items()
                            ]
                        ) and all( k in result["scenario"] for k in scenario)
                        if matches:
                            return True
                except Exception as e:
                    pass
        return False

    def time_case(self, case):
        elapsed = None
        timedout = False
        with self.elapsed_timer() as t:
            try:
                with timeout(self.test_timeout, RuntimeError):
                    case()
                    elapsed = t()
            except RuntimeError:
                timedout = True
        return {"time": elapsed, "timedout": timedout}

    def run_cases(self, run_case_fn, cases, scenario):
        for case in cases:
            results = {"results": []}
            if self.case_already_ran(case, scenario):
                print(f"Skipping already run case: {case}")
                continue
            else:
                print(f"Running case: {case}")
                result = self.time_case(partial(run_case_fn, case))
                result["end_time"] = str(datetime.datetime.now())
                result["scenario"] = scenario
                results["results"].append({**case, **result})

            json_results = {"results": []}
            if os.path.exists(self.out_file):
                with open(self.out_file, "r") as f:
                    try:
                        json_results = json.loads(f.read())
                    except Exception as e:
                        pass
            json_results["results"] += results["results"]
            with open(self.out_file, "w") as f:
                json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    unittest.main()
