from pysmt.shortcuts import And, Symbol

from funman.search.episode import BoxSearchEpisode
from funman.search.search import Search


class BottomUpBoxSearch(Search):
    def search(
        self, problem: "AnalysisScenario", config: "FUNMANConfig"
    ) -> ParameterSpace:
        episode = BoxSearchEpisode(config=config, problem=problem)

        result = self.find_box(episode)

        parameter_space = ParameterSpace(
            true_boxes=[result],
            false_boxes=[],
            true_points=[],
            false_points=[],
        )
        return parameter_space

    def _setup_ea(self, solver, episode):
        """


        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """
        solver.push(1)
        parameters = episode.problem.parameters
        parameter_symbols = [Symbol(p.name, REAL) for p in parameters]
        box_bounds_formula = And(
            [
                And(
                    LE(Symbol(f"{p.name}_lb", REAL), Symbol(p.name, REAL)),
                    LT(Symbol(p.name, REAL), Symbol(f"{p.name}_ub", REAL)),
                )
                for p in parameters
            ]
        )
        forall_formula = ForAll(
            parameter_symbols,
            And(
                episode.problem._assume_model,
                Not(episode.problem._assume_query),
            ),
        )

        formula = And(box_bounds_formula, forall_formula)

        episode._formula_stack.append(formula)
        solver.add_assertion(formula)

    def find_box(episode):
        if episode.config.solver == "dreal":
            opts = {
                "dreal_precision": episode.config.dreal_precision,
                "dreal_log_level": episode.config.dreal_log_level,
                "dreal_mcts": episode.config.dreal_mcts,
            }
        else:
            opts = {}
        with Solver(
            name=episode.config.solver,
            logic=QF_NRA,
            solver_options=opts,
        ) as solver:
            self._initialize_encoding(solver, episode)
            self._setup_ea(solver, episode)
            # if episode.config.save_smtlib:
            #     self.store_smtlib(
            #         episode, box, filename=f"tp_{episode._iteration}.smt2"
            #     )
            res1 = None
            if solver.solve():
                res1 = solver.get_model()
            return res1
