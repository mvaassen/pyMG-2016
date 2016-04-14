# coding=utf-8
import scipy.sparse as sp
import scipy.sparse.linalg as spLA

from pymg.smoother_base import SmootherBase


class GaussSeidel(SmootherBase):
    """Implementation of the Gauss-Seidel iteration

    Attributes:
        P (scipy.sparse.csc_matrix): Preconditioner
        Pinv (scipy.sparse.csc_matrix): inverse of the Preconditioner
    """

    def __init__(self, A, *args, **kwargs):
        """Initialization routine for the smoother

        Args:
            A (scipy.sparse.csc_matrix): sparse matrix A of the system to solve
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super(GaussSeidel, self).__init__(A, *args, **kwargs)

        self.D = sp.spdiags(self.A.diagonal(), 0, self.A.shape[0], self.A.shape[1],
                            format='csc')
        self.L = sp.tril(self.A, -1, format='csc')
        self.U = sp.triu(self.A, 1, format='csc')
        # precompute inverse of the preconditioner for later usage
        self.Pinv = spLA.inv(self.D + self.L)

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = self.Pinv.dot(rhs - self.U.dot(u_old))
        return u_new
