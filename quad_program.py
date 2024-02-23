from cvxopt import matrix, solvers
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class quad_program_for_perching:

    # quad_program_for_perching
    ## Dependencies
    #- numpy
    #- cvxopt \n

    ## Descirptions

    ## Notations
    #In this class we use the cvxopt, the basic quadratic problem can be solved as
   # \n \n
    #min  x 1/2 * x^T Q x + p^T x \n
    #subject to   G x <= h \n
    #A x == b \n

    #Then one can use:
    #```Python



    mu_vel = 1
    mu_acc = 1
    mu_jerk = 1
    mu_snap = 1

    def __init__(self, polynomial_order, T_end):
        """
        We need to obtain the coefficients due to different order derivatives. \n
        Here we define a unit polynomial as
        /
        X**2 + X + 1,
        /
        that is all the coefficient as one, \n
        this is quite representive, because we barely can multiply the corresponding  \n
        coefficients and get any polynomial we want. \n
        """
        self.t = sp.Symbol('t')
        self.Tend = T_end
        self.poly_order = polynomial_order
        # It is a time symbol, since the polynomial is a function of time, this is just
        # all the symbol we need.

        marker_for_unit_polynomial = np.eye(polynomial_order + 1)

        list_of_unit_polynomials = []
        # Here we initialize them.
        for i in range(polynomial_order + 1):
            single_polynomial = self.t ** (polynomial_order - i)
            list_of_unit_polynomials.append(single_polynomial)

        # horizontal_polynomial_mat shown as  [[Poly(1.0*t**2, t, domain='RR') Poly(1.0*t, t, domain='RR')
        # Poly(1.0, t, domain='RR')]] that is, it is a nx1 vector indeed
        self.pos_horizontal_polynomial_mat = np.mat(list_of_unit_polynomials)
        # print('self. pos_horizontal_polynomial_mat:', self. pos_horizontal_polynomial_mat)
        self.vel_horizontal_polynomial_mat = self.diff_mat(self.pos_horizontal_polynomial_mat)
        # print('self. vel_horizontal_polynomial_mat:', self. vel_horizontal_polynomial_mat)
        self.acc_horizontal_polynomial_mat = self.diff_mat(self.vel_horizontal_polynomial_mat)
        self.jerk_horizontal_polynomial_mat = self.diff_mat(self.acc_horizontal_polynomial_mat)
        self.snap_horizontal_polynomial_mat = self.diff_mat(self.jerk_horizontal_polynomial_mat)

        # print('self. pos_horizontal_polynomial_mat:', self. pos_horizontal_polynomial_mat)
        # print('self. snap_horizontal_polynomial_mat:', self. snap_horizontal_polynomial_mat)
        self.int_vel_square_mat = self.Calc_SQ(self.vel_horizontal_polynomial_mat)
        self.int_acc_square_mat = self.Calc_SQ(self.acc_horizontal_polynomial_mat)
        self.int_jerk_square_mat = self.Calc_SQ(self.jerk_horizontal_polynomial_mat)
        self.int_snap_square_mat = self.Calc_SQ(self.snap_horizontal_polynomial_mat)

        # print('self. vel_square_mat:',self. vel_square_mat)
        self.int_total_square_mat = self.mu_vel * self.int_vel_square_mat + \
                                    self.mu_acc * self.int_acc_square_mat + \
                                    self.mu_jerk * self.int_jerk_square_mat + \
                                    self.mu_snap * self.int_snap_square_mat

        #print('self. int_total_square_mat:', self.int_total_square_mat)

        self.Q = matrix(
            self.subs_mat(self.int_total_square_mat, self.Tend) - self.subs_mat(self.int_total_square_mat, 0), tc='d')
        self.p = matrix([0.0] * (polynomial_order + 1), tc='d')

        self.A = []
        self.b = []
        self.G = []
        self.h = []

    def set_Tend(self, T_end):
        self.Tend = T_end
        self.Q = matrix(
            self.subs_mat(self.int_total_square_mat, self.Tend) - self.subs_mat(self.int_total_square_mat, 0), tc='d')

    def add_ob_at_t(self, ob_p, mu, t):
        pos_mat = self.subs_mat(self.pos_horizontal_polynomial_mat, t)
        Q_add = - matrix(pos_mat.T * pos_mat)  # * (self.Tend - self.t) / self.Tend
        p_add = matrix(ob_p * pos_mat.T, tc='d')  # * (self.Tend - self.t) / self.Tend
        self.Q = self.Q + mu * Q_add
        self.p = self.p + mu * p_add

    def add_att_at_t(self, att_p, mu, t):
        pos_mat = self.subs_mat(self.pos_horizontal_polynomial_mat, t)
        Q_add = matrix(pos_mat.T * pos_mat)  # * (self.Tend - self.t) / self.Tend

        p_add = -   matrix(att_p * pos_mat.T, tc='d')  # * (self.Tend - self.t) / self.Tend

        # print('p_add:', p_add)
        self.Q = self.Q + mu * Q_add
        self.p = self.p + mu * p_add

    def set_coeff(self, m_vel, m_acc, m_jerk, m_snap):
        if m_vel is not None:
            self.mu_vel = m_vel
        if m_acc is not None:
            self.mu_acc = m_acc
        if m_jerk is not None:
            self.mu_jerk = m_jerk
        if m_snap is not None:
            self.mu_snap = m_snap

        self.int_total_square_mat = self.mu_vel * self.int_vel_square_mat + \
                                    self.mu_acc * self.int_acc_square_mat + \
                                    self.mu_jerk * self.int_jerk_square_mat + \
                                    self.mu_snap * self.int_snap_square_mat
        # print('self. int_total_square_mat:',self. int_total_square_mat)

        self.Q = matrix(
            self.subs_mat(self.int_total_square_mat, self.Tend) - self.subs_mat(self.int_total_square_mat, 0), tc='d')
        # print('Q at set:', self.Q)

    def Calc_SQ(self, horizontal_vec):
        Transpose_mul_origin_Mat = horizontal_vec.T @ horizontal_vec
        Int_Mat = self.Indef_int_mat(Transpose_mul_origin_Mat)
        return Int_Mat

    def clear_constraints(self):
        self.A = []
        self.b = []
        self.G = []
        self.h = []

    def add_pos_constraint_1_by_1(self, tic, pos):
        unit_pos_at_tic = self.subs_mat(self.pos_horizontal_polynomial_mat, tic)
        self.A.append(unit_pos_at_tic[0, :].tolist())
        self.b.append(pos)

    def add_upper_pos_constraint_1_by_1(self, tic, pos):
        unit_pos_at_tic = self.subs_mat(self.pos_horizontal_polynomial_mat, tic)
        self.G.append(unit_pos_at_tic[0, :].tolist())
        self.h.append(pos)

    def add_lower_pos_constraint_1_by_1(self, tic, pos):
        unit_pos_at_tic = self.subs_mat(self.pos_horizontal_polynomial_mat, tic)
        self.G.append((-unit_pos_at_tic[0, :]).tolist())
        self.h.append(-pos)

    def add_symmetry_pos_constraint_1_by_1(self, tic, pos):
        self.add_upper_pos_constraint_1_by_1(tic, pos)
        self.add_lower_pos_constraint_1_by_1(tic, pos)

    def add_vel_constraint_1_by_1(self, tic, vel):
        unit_vel_at_tic = self.subs_mat(self.vel_horizontal_polynomial_mat, tic)
        self.A.append(unit_vel_at_tic[0, :].tolist())
        # print('unit_vel_at_tic[0,:].tolist():',unit_vel_at_tic[0,:].tolist())
        self.b.append(vel)

    def add_upper_vel_constraint_1_by_1(self, tic, vel):
        unit_vel_at_tic = self.subs_mat(self.vel_horizontal_polynomial_mat, tic)
        self.G.append(unit_vel_at_tic[0, :].tolist())
        self.h.append(vel)

    def add_lower_vel_constraint_1_by_1(self, tic, vel):
        unit_vel_at_tic = self.subs_mat(self.vel_horizontal_polynomial_mat, tic)
        self.G.append((-unit_vel_at_tic[0, :]).tolist())
        self.h.append(-vel)

    def add_symmetry_vel_constraint_1_by_1(self, tic, vel_abs):
        self.add_upper_vel_constraint_1_by_1(tic, vel_abs)
        self.add_lower_vel_constraint_1_by_1(tic, -vel_abs)

    def add_acc_constraint_1_by_1(self, tic, acc):
        unit_acc_at_tic = self.subs_mat(self.acc_horizontal_polynomial_mat, tic)
        self.A.append(unit_acc_at_tic[0, :].tolist())
        # print('unit_acc_at_tic[0,:].tolist():',unit_acc_at_tic[0,:].tolist())
        self.b.append(acc)

    def add_upper_acc_constraint_1_by_1(self, tic, acc):
        unit_acc_at_tic = self.subs_mat(self.acc_horizontal_polynomial_mat, tic)
        self.G.append(unit_acc_at_tic[0, :].tolist())
        self.h.append(acc)

    def add_lower_acc_constraint_1_by_1(self, tic, acc):
        unit_acc_at_tic = self.subs_mat(self.acc_horizontal_polynomial_mat, tic)
        self.G.append((-unit_acc_at_tic[0, :]).tolist())
        self.h.append(-acc)

    def add_symmetry_acc_constraint_1_by_1(self, tic, acc_abs):
        self.add_upper_acc_constraint_1_by_1(tic, acc_abs)
        self.add_lower_acc_constraint_1_by_1(tic, -acc_abs)

    def add_jerk_constraint_1_by_1(self, tic, jerk):
        unit_jerk_at_tic = self.subs_mat(self.jerk_horizontal_polynomial_mat, tic)
        self.A.append(unit_jerk_at_tic[0, :].tolist())
        # print('unit_jerk_at_tic[0,:].tolist():',unit_jerk_at_tic[0,:].tolist())
        self.b.append(jerk)

    def opt(self):
        # solvers.options['show_progress'] = False
        A_mat = matrix(np.mat(self.A), tc='d')
        b_mat = matrix(np.mat(self.b).T, tc='d')

        #print('Q:', self.Q)
        #print('p:', self.p)
        #print('A_mat:', A_mat)
        #print('b_mat:', b_mat)
        if not len(self.G):
            sol = solvers.qp(self.Q, self.p, None, None, A=A_mat, b=b_mat)
        else:
            G_mat = matrix(np.mat(self.G), tc='d')
            h_mat = matrix(np.mat(self.h).T, tc='d')

            sol = solvers.qp(self.Q, self.p, G_mat, h_mat, A_mat, b_mat)

        return sol['x']

    def diff_mat(self, symbol_mat):
        [row, col] = symbol_mat.shape
        t = sp.Symbol('t')
        output_mat = np.tile(t, (row, col))

        for i in range(row):
            for j in range(col):
                output_mat[i, j] = sp.diff(symbol_mat[i, j], self.t)
        return output_mat

    def Indef_int_mat(self, symbol_mat):
        [row, col] = symbol_mat.shape
        t = sp.Symbol('t')
        output_mat = np.tile(t, (row, col))
        # t = sp.Symbol('t')
        [row, col] = output_mat.shape
        for i in range(row):
            for j in range(col):
                output_mat[i, j] = sp.integrate(symbol_mat[i, j], self.t)
        return output_mat

    def subs_mat(self, the_mat, value):
        # output_mat = symbol_mat
        [row, col] = the_mat.shape
        # t = sp.Symbol('t')
        output_mat = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                output_mat[i, j] = the_mat[i, j].subs(self.t, value)
        return output_mat

    def g_of_t(self,x):
        coefficients = (x)  # 获取系数
        terms = []
        for i in range(len(coefficients) - 1, -1, -1):
            if coefficients[i] != 0:  # 非零系数才添加
                term = f"{coefficients[len(coefficients)-1-i]}*t^{i}" if i > 0 else f"{coefficients[i]}"
                terms.append(term)

        return " + ".join(terms)

    def plot_g(self, x, num_points=1000):
        """
        Plot the function g(t) over the interval [0, Tend].

        :param x: Coefficients of the polynomial.
        :param num_points: Number of points to plot.
        """
        t_values = np.linspace(0, self.Tend, num_points)
        g_values = [self.evaluate_g(x, t) for t in t_values]

        plt.figure(figsize=(10, 5))
        plt.plot(t_values, g_values, label="g(t)")
        plt.title("Visualization of g(t)")
        plt.xlabel("t")
        plt.ylabel("g(t)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def evaluate_g(self, x, t_value):
        """
        Evaluate the polynomial g at a specific time t.

        :param x: Coefficients of the polynomial.
        :param t_value: The time at which to evaluate g.
        :return: The value of g at time t.
        """
        t = sp.Symbol('t')
        g_expression = self.g_of_t(x)
        g_function = sp.lambdify(t, g_expression, 'numpy')
        return g_function(t_value)



