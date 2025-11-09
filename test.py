import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from run_exp_test import run_inference_for_optimization


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=2,
                         xu=9, type_var=np.int,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = run_inference_for_optimization(x[0],x[1])
      #  f2 = (x[0] - 1) ** 2 + x[1] ** 2

       # g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
       # g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1]
        #out["G"] = [g1, g2]


problem = MyProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()