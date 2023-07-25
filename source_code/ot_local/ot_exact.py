import numpy as np
import ot


def fpbm_new(distances):
    import gurobipy as gp
    from gurobipy import GRB

    # return fpbm_new2(distances)
    m = gp.Model("fpbm")

    d1 = distances.shape[0]
    d2 = distances.shape[1]
    w_i = 1.0 / float(d1)
    w_j = 1.0 / float(d2)
    flow_indices = [(i, j) for i in range(d1) for j in range(d2)]
    f = m.addVars(flow_indices, vtype=GRB.CONTINUOUS, lb=0.0, ub=min(w_i, w_j), name="f")

    # constraints
    m.addConstrs((gp.quicksum(f[i, j] for j in range(d2)) == w_i for i in range(d1)))
    m.addConstrs((gp.quicksum(f[i, j] for i in range(d1)) == w_j for j in range(d2)))

    # objective
    expr_list = [f[i, j] * distances[i, j] for i in range(d1) for j in range(d2)]
    m.setObjective(gp.quicksum(expr_list))

    # optimization
    m.optimize()

    return m.ObjVal

def emd(distances):
    d1 = distances.shape[0]
    d2 = distances.shape[1]
    w_i = np.array([1.0 / float(d1) for _ in range(d1)])
    w_j = np.array([1.0 / float(d2) for _ in range(d2)])
    return np.sum(np.multiply(ot.emd(w_i, w_j, distances), distances))


if __name__ == "__main__":
    d = np.random.random((2000, 3000))
    print(d.shape)
    # print(fpbm_new(d))
    print(emd(d))