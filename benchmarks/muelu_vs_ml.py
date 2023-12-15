import numpy as np
from PyTrilinos import Epetra, AztecOO, Teuchos
from PyTrilinos import ML, MueLu

def create_elasticity_matrix(mesh_size):
    # a simple 2D elasticity matrix for a square mesh (a sparse coefficient matrix K corresponding to Ku = F)
    # TODO: modify this on the basis of some real world benchmark data on elasticity

    n_nodes = (mesh_size + 1) ** 2
    mesh_spacing = 1.0 / mesh_size

    matrix = Epetra.CrsMatrix(Epetra.Copy)

    for i in range(n_nodes):
        row = i
        matrix.InsertGlobalValues(row, [row], [4.0])  # Diagonal term

        if i % (mesh_size + 1) != 0:
            matrix.InsertGlobalValues(row, [row - 1], [-1.0])  # Left neighbor
        if (i + 1) % (mesh_size + 1) != 0:
            matrix.InsertGlobalValues(row, [row + 1], [-1.0])  # Right neighbor
        if i >= (mesh_size + 1):
            matrix.InsertGlobalValues(row, [row - (mesh_size + 1)], [-1.0])  # Top neighbor
        if i < n_nodes - (mesh_size + 1):
            matrix.InsertGlobalValues(row, [row + (mesh_size + 1)], [-1.0])  # Bottom neighbor

    matrix.FillComplete()

    return matrix

def create_rhs_vector(mesh_size):
    # Create a simple right-hand side vector
    rhs = Epetra.Vector(Epetra.Map(mesh_size**2, 0))

    # Fill the vector with some values
    rhs[:] = 1.0

    return rhs

def solve_with_muelu(matrix, rhs):
    comm = matrix.Comm()
    muelu_params = Teuchos.ParameterList()
    
    # Set MueLu parameters as needed
    muelu_params.set("verbosity", "low")
    muelu_params.set("max levels", 10)

    muelu_prec = MueLu.CreateTpetraPreconditioner(matrix, muelu_params)
    
    solver = AztecOO.AztecOO(matrix, lhs=rhs, prec=muelu_prec)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg)
    solver.Iterate(1000, 1e-10)

    return solver.NumIters()

def solve_with_ml(matrix, rhs):
    comm = matrix.Comm()
    ml_params = Teuchos.ParameterList()
    ml_params.set("ML output", 10)
    ml_params.set("max levels", 10)

    ml_prec = ML.MultiLevelPreconditioner(matrix, False)
    ml_prec.setParameterList(ml_params)
    ml_prec.ComputePreconditioner()

    solver = AztecOO.AztecOO(matrix, lhs=rhs, prec=ml_prec)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg)
    solver.Iterate(1000, 1e-10)

    return solver.NumIters()

def benchmark_solver(matrix, rhs, solver_function, num_trials=5):
    times = []
    for _ in range(num_trials):
        start_time = Teuchos.Time()
        iterations = solver_function(matrix, rhs)
        end_time = Teuchos.Time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    average_time = np.mean(times)
    print(f"Average Iteration Time: {average_time} seconds")
    print("\n")
    print(f"Average Number of Iterations: {iterations}")

if __name__ == "__main__":
    mesh_size = 64

    matrix = create_elasticity_matrix(mesh_size)
    rhs = create_rhs_vector(mesh_size)

    print("Benchmarking MueLu:")
    benchmark_solver(matrix, rhs, solve_with_muelu)
    print("\n")
    print("Benchmarking ML:")
    benchmark_solver(matrix, rhs, solve_with_ml)

